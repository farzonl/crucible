#!/usr/bin/env python3
"""Select a bounded CTS baseline, run it, and record QPA results for Pages."""

import argparse
import hashlib
import json
import os
import signal
import shlex
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STATUSES = (
    "Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning",
    "InternalError", "ResourceError", "Crash", "Timeout", "Pending", "NotRun",
    "CompilerError", "MetadataError", "AdapterUnsupported", "NotExercised",
)
FAILURES = {"Fail", "InternalError", "ResourceError", "Crash", "Timeout",
            "CompilerError", "MetadataError"}
TRACE_EVENTS = {"attempt", "compiled", "failed", "adapter_failed", "metadata_failed"}


def read_cases(path: Path, active: frozenset[Path] = frozenset()) -> list[str]:
    path = path.resolve()
    if path in active:
        raise ValueError(f"Cyclic caselist include: {path}")
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("dEQP-VK."):
            if any(char.isspace() for char in line) or "*" in line:
                raise ValueError(f"Expected an exact CTS case, got {line!r}")
            cases.append(line)
        else:
            cases.extend(read_cases(path.parent / line, active | {path}))
    return cases


def select_cases(source: Path, prefix: str, limit: int) -> list[str]:
    if limit < 1:
        raise ValueError("Case limit must be positive")
    candidates = sorted(set(name for name in read_cases(source)
                            if name.startswith(prefix)))
    if not candidates:
        raise ValueError(f"No CTS cases match {prefix}")
    count = min(limit, len(candidates))
    return [candidates[index * len(candidates) // count] for index in range(count)]


def parse_qpa(path: Path, planned: list[str]) -> list[dict]:
    if not planned or len(set(planned)) != len(planned):
        raise ValueError("Planned caselist must be nonempty and unique")
    expected = set(planned)
    results = {}
    current = None
    fragment = []

    def finish(code: str, details: str) -> None:
        if current not in expected:
            raise ValueError(f"Unexpected QPA test: {current}")
        if current in results:
            raise ValueError(f"Duplicate QPA test: {current}")
        if code not in STATUSES:
            raise ValueError(f"Unknown CTS status: {code}")
        results[current] = {"name": current, "code": code, "details": details}

    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if line.startswith("#beginTestCaseResult"):
                args = shlex.split(line)
                if current is not None or len(args) != 2:
                    raise ValueError("Invalid QPA begin record")
                current = args[1]
                fragment = []
            elif line.startswith("#endTestCaseResult"):
                if current is None:
                    raise ValueError("QPA end without begin")
                root = ET.fromstring("".join(fragment))
                items = list(root.iter("Result"))
                if len(items) != 1:
                    raise ValueError(f"Expected one Result for {current}")
                finish(items[0].attrib["StatusCode"], "".join(items[0].itertext()).strip())
                current = None
            elif line.startswith("#terminateTestCaseResult"):
                if current is None:
                    raise ValueError("QPA termination without begin")
                details = " ".join(shlex.split(line)[1:])
                code = details if details in STATUSES else (
                    "Timeout" if details == "Watchdog timeout occurred." else "Crash")
                finish(code, details)
                current = None
            elif current is not None:
                fragment.append(line)
    if current is not None:
        finish("InternalError", "Incomplete QPA result: run interrupted")
    if not results:
        raise ValueError("QPA contains no test results; refusing to publish a score")
    return [results.get(name, {"name": name, "code": "NotRun",
                              "details": "No result in this bounded run"})
            for name in planned]


def summarize(tests: list[dict]) -> dict:
    counts = Counter(test["code"] for test in tests)
    total = len(tests)
    return {
        "total": total, "passed": counts["Pass"],
        "failed": sum(counts[code] for code in FAILURES),
        "unsupported": counts["NotSupported"] + counts["AdapterUnsupported"],
        "warnings": counts["QualityWarning"] + counts["CompatibilityWarning"],
        "notRun": counts["NotRun"] + counts["Pending"] + counts["NotExercised"],
        "passRate": round(100 * counts["Pass"] / total, 2) if total else None,
        "counts": {code: counts[code] for code in STATUSES},
    }


def build_tree(tests: list[dict]) -> dict:
    root = {"name": "dEQP-VK", "path": "dEQP-VK", "children": {}}
    for test in tests:
        node = root
        parts = test["name"].split(".")[1:]
        for index, part in enumerate(parts):
            children = node["children"]
            if part not in children:
                children[part] = {
                    "name": part, "path": ".".join(["dEQP-VK", *parts[:index + 1]]),
                    "children": {},
                }
            node = children[part]
        if node["children"] or "test" in node:
            raise ValueError(f"Conflicting case path: {test['name']}")
        node["test"] = test

    def aggregate(node: dict) -> list[dict]:
        children = node.pop("children")
        descendants = []
        if "test" in node:
            if children:
                raise ValueError(f"Case is also a directory: {node['path']}")
            descendants.append(node.pop("test"))
            node.update(descendants[0])
        else:
            node["children"] = [children[key] for key in sorted(children)]
            for child in node["children"]:
                descendants.extend(aggregate(child))
        node.update(summarize(descendants))
        return descendants

    aggregate(root)
    return root


def compiler_results(tests: list[dict], evidence: dict) -> list[dict]:
    unexpected = set(evidence) - {test["name"] for test in tests}
    if unexpected:
        raise ValueError(f"Compiler trace contains unexpected cases: {unexpected}")
    for test in tests:
        events = evidence.get(test["name"], {})
        if set(events) - TRACE_EVENTS or any(
                not isinstance(count, int) or count < 0 for count in events.values()):
            raise ValueError(f"Invalid compiler trace for {test['name']}")
        test.update(ctsCode=test["code"], compiler=events)
        code = None
        if events.get("adapter_failed"):
            code = "AdapterUnsupported"
        elif events.get("metadata_failed"):
            code = "MetadataError"
        elif events.get("failed"):
            code = "CompilerError"
        elif test["code"] == "Pass" and (
                not events.get("attempt") or
                events.get("compiled", 0) != events["attempt"]):
            code = "NotExercised"
        if code:
            test["details"] = (f"{code}: CTS reported {test['ctsCode']}; "
                               f"compiler trace {events}. {test['details']}")
            test["code"] = code
    return tests


def record(qpa: Path, caselist: Path, metadata: dict, data_dir: Path,
           cache: Path | None = None, reference_qpa: Path | None = None,
           reference_metadata: dict | None = None) -> dict:
    if metadata.get("measurementMode") != "crucible":
        raise ValueError("Refusing to publish native CTS as a Crucible compiler score")
    planned = read_cases(caselist)
    tests = compiler_results(parse_qpa(qpa, planned), metadata["compilerEvidence"])
    if not any(events.get("attempt") for events in metadata["compilerEvidence"].values()):
        raise ValueError("No Crucible shader compilation was observed; refusing to publish a score")
    caselist_hash = hashlib.sha256(caselist.read_bytes()).hexdigest()
    reference = None
    reference_tests = {}
    if reference_qpa is not None:
        if reference_metadata is None or reference_metadata.get("measurementMode") != "reference":
            raise ValueError("Native reference metadata is required")
        for key in ("sha", "ctsRevision", "runtimeVersion"):
            if reference_metadata[key] != metadata[key]:
                raise ValueError(f"Reference does not match compiler run: {key}")
        if reference_metadata["caselistSha256"] != caselist_hash:
            raise ValueError("Reference caselist does not match compiler run")
        native = parse_qpa(reference_qpa, planned)
        reference = {**summarize(native), "tree": build_tree(native), "tests": native,
                     "timedOut": reference_metadata["timedOut"]}
        reference_tests = {test["name"]: test for test in native}
        for test in tests:
            test["referenceCode"] = reference_tests[test["name"]]["code"]
    target = data_dir / "vulkan-cts"
    previous = {}
    for history_path in (target / "history.json", cache):
        if history_path is None or not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError("CTS history must be a JSON list")
        for old in history:
            if old.get("measurementMode") != "crucible":
                continue
            if (old["sha"] not in previous or
                    old["timestamp"] > previous[old["sha"]]["timestamp"]):
                previous[old["sha"]] = old
    entry = {
        **{key: value for key, value in metadata.items() if key != "compilerEvidence"},
        "schemaVersion": 2,
        "shortSha": metadata["sha"][:7],
        "caselistSha256": caselist_hash,
        "measurement": "CTS executes Crucible-generated MSL through MoltenVK; no native fallback",
        **summarize(tests),
    }
    if reference:
        entry["reference"] = {key: value for key, value in reference.items()
                              if key not in ("tree", "tests")}
    history = [old for old in previous.values() if old["sha"] != entry["sha"]]
    history.append(entry)
    history.sort(key=lambda old: old["timestamp"])
    latest = {**entry, "tests": tests, "tree": build_tree(tests),
              "failures": [test for test in tests if test["code"] in FAILURES]}
    if reference:
        latest["reference"] = reference
    target.mkdir(parents=True, exist_ok=True)
    for name, value in (("history.json", history), ("latest.json", latest),
                        ("badge.json", {
                            "schemaVersion": 1, "label": "Vulkan CTS (Crucible)",
                            "message": f"{entry['passRate']}%",
                            "color": "green" if entry["passRate"] >= 85 else "orange",
                        })):
        (target / name).write_text(json.dumps(value), encoding="utf-8")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(history), encoding="utf-8")
    return latest


def execute_case(command: list[str], environment: dict[str, str], timeout: float) -> int | None:
    # Killing the process group also terminates any in-flight compiler child.
    with subprocess.Popen(command, env=environment, start_new_session=True) as process:
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return None


def run_cases(args: argparse.Namespace) -> dict:
    planned = read_cases(args.caselist)
    args.qpa.parent.mkdir(parents=True, exist_ok=True)
    args.qpa.unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.pop("CRUCIBLE_CTS_COMPILER", None)
    environment.pop("CRUCIBLE_CTS_TRACE", None)
    evidence = {}
    metadata = {
        "sha": args.sha, "runUrl": args.run_url,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ctsRevision": args.cts_revision, "runtimeVersion": args.runtime_version,
        "caselistSha256": hashlib.sha256(args.caselist.read_bytes()).hexdigest(),
        "baseline": args.baseline, "platform": "macOS / MoltenVK",
        "measurementMode": "crucible" if args.compiler else "reference",
        "compilerEvidence": evidence, "timeoutSeconds": args.timeout,
        "adapterVersion": hashlib.sha256(
            Path(__file__).with_name("cts_moltenvk_hook.h").read_bytes() +
            Path(__file__).with_name("prepare_cts_moltenvk.py").read_bytes()).hexdigest(),
        "compiler": str(args.compiler.resolve()) if args.compiler else None,
        "shaderTimeoutSeconds": 30,
        "exitCode": 0, "timedOut": False,
    }
    if args.compiler:
        environment["CRUCIBLE_CTS_COMPILER"] = str(args.compiler.resolve(strict=True))
    deadline = time.monotonic() + args.timeout
    case_dir = args.qpa.parent / ("crucible-cases" if args.compiler else "reference-cases")
    case_dir.mkdir(exist_ok=True)
    try:
        for index, name in enumerate(planned):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                metadata["timedOut"] = True
                break
            case_qpa = case_dir / f"{index}.qpa"
            trace = case_dir / f"{index}.trace"
            case_qpa.unlink(missing_ok=True)
            trace.unlink(missing_ok=True)
            if args.compiler:
                environment["CRUCIBLE_CTS_TRACE"] = str(trace.resolve())
            command = [
                str(args.binary.resolve()), f"--deqp-case={name}",
                f"--deqp-archive-dir={args.archive.resolve()}",
                f"--deqp-vk-library-path={args.library.resolve()}",
                f"--deqp-log-filename={case_qpa.resolve()}",
                "--deqp-log-images=disable", "--deqp-log-shader-sources=disable",
            ]
            metadata["invocation"] = shlex.join(command).replace(name, "<selected-case>")
            exit_code = execute_case(command, environment, remaining)
            metadata["exitCode"] = exit_code if exit_code != 0 else metadata["exitCode"]
            if exit_code is None:
                metadata["timedOut"] = True
                print(f"CTS budget exceeded during {name}")
                if not case_qpa.exists() or "#beginTestCaseResult" not in case_qpa.read_text():
                    case_qpa.write_text(
                        f"#beginTestCaseResult {name}\n#terminateTestCaseResult Timeout\n",
                        encoding="utf-8")
            if not case_qpa.exists():
                raise RuntimeError(f"CTS produced no QPA for {name}; exit {exit_code}")
            parsed = parse_qpa(case_qpa, [name])
            with args.qpa.open("a", encoding="utf-8") as combined:
                # Terminate an interrupted record before appending the next session.
                text = case_qpa.read_text(encoding="utf-8", errors="replace")
                combined.write(text + "\n")
                if parsed[0]["details"] == "Incomplete QPA result: run interrupted":
                    combined.write("#terminateTestCaseResult InternalError\n")
            if args.compiler:
                events = Counter(trace.read_text().splitlines()) if trace.exists() else Counter()
                if set(events) - TRACE_EVENTS:
                    raise ValueError(f"Unknown compiler trace events for {name}")
                evidence[name] = dict(events)
            if metadata["timedOut"]:
                break
    finally:
        args.metadata.write_text(json.dumps(metadata), encoding="utf-8")
    if not args.qpa.exists():
        raise RuntimeError("CTS produced no QPA output")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    select = commands.add_parser("select")
    select.add_argument("--source", required=True, type=Path)
    select.add_argument("--prefix", default="dEQP-VK.spirv_assembly.")
    select.add_argument("--limit", default=256, type=int)
    select.add_argument("--output", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--binary", required=True, type=Path)
    run.add_argument("--compiler", type=Path, help="Enable Crucible; omit only for the native reference")
    run.add_argument("--archive", required=True, type=Path)
    run.add_argument("--library", required=True, type=Path)
    run.add_argument("--caselist", required=True, type=Path)
    run.add_argument("--qpa", required=True, type=Path)
    run.add_argument("--metadata", required=True, type=Path)
    run.add_argument("--timeout", default=900, type=int)
    run.add_argument("--sha", required=True)
    run.add_argument("--run-url", required=True)
    run.add_argument("--cts-revision", required=True)
    run.add_argument("--runtime-version", required=True)
    run.add_argument("--baseline", required=True)
    save = commands.add_parser("record")
    save.add_argument("--qpa", required=True, type=Path)
    save.add_argument("--caselist", required=True, type=Path)
    save.add_argument("--metadata", required=True, type=Path)
    save.add_argument("--data-dir", required=True, type=Path)
    save.add_argument("--history-cache", type=Path)
    save.add_argument("--reference-qpa", required=True, type=Path)
    save.add_argument("--reference-metadata", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "select":
        cases = select_cases(args.source, args.prefix, args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(cases) + "\n", encoding="utf-8")
        print(f"Selected {len(cases)} cases")
    elif args.command == "run":
        if args.timeout < 1:
            parser.error("--timeout must be positive")
        metadata = run_cases(args)
        if metadata["exitCode"] not in (0, None):
            print(f"CTS exited with {metadata['exitCode']}; recording actual QPA results")
    else:
        latest = record(args.qpa, args.caselist,
                        json.loads(args.metadata.read_text(encoding="utf-8")),
                        args.data_dir, args.history_cache, args.reference_qpa,
                        json.loads(args.reference_metadata.read_text(encoding="utf-8")))
        print(f"CTS: {latest['passed']}/{latest['total']} passed; "
              f"{latest['failed']} failed, {latest['notRun']} unrun")


if __name__ == "__main__":
    main()
