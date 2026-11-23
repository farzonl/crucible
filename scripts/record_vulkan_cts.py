#!/usr/bin/env python3
"""Select a bounded CTS baseline, run it, and record QPA results for Pages."""

import argparse
import hashlib
import json
import shlex
import subprocess
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

STATUSES = (
    "Pass", "Fail", "NotSupported", "QualityWarning", "CompatibilityWarning",
    "InternalError", "ResourceError", "Crash", "Timeout", "Pending", "NotRun",
)
FAILURES = {"Fail", "InternalError", "ResourceError", "Crash", "Timeout"}


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
        "unsupported": counts["NotSupported"],
        "warnings": counts["QualityWarning"] + counts["CompatibilityWarning"],
        "notRun": counts["NotRun"] + counts["Pending"],
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


def record(qpa: Path, caselist: Path, metadata: dict, data_dir: Path,
           cache: Path | None = None) -> dict:
    tests = parse_qpa(qpa, read_cases(caselist))
    target = data_dir / "vulkan-cts"
    previous = {}
    for history_path in (target / "history.json", cache):
        if history_path is None or not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            raise ValueError("CTS history must be a JSON list")
        for old in history:
            if (old["sha"] not in previous or
                    old["timestamp"] > previous[old["sha"]]["timestamp"]):
                previous[old["sha"]] = old
    entry = {
        **metadata, "schemaVersion": 1,
        "shortSha": metadata["sha"][:7],
        "caselistSha256": hashlib.sha256(caselist.read_bytes()).hexdigest(),
        "measurement": "Native MoltenVK Vulkan CTS; does not invoke Crucible",
        **summarize(tests),
    }
    history = [old for old in previous.values() if old["sha"] != entry["sha"]]
    history.append(entry)
    history.sort(key=lambda old: old["timestamp"])
    latest = {**entry, "tests": tests, "tree": build_tree(tests),
              "failures": [test for test in tests if test["code"] in FAILURES]}
    target.mkdir(parents=True, exist_ok=True)
    for name, value in (("history.json", history), ("latest.json", latest),
                        ("badge.json", {
                            "schemaVersion": 1, "label": "Vulkan CTS (MoltenVK)",
                            "message": f"{entry['passRate']}%",
                            "color": "green" if entry["passRate"] >= 85 else "orange",
                        })):
        (target / name).write_text(json.dumps(value), encoding="utf-8")
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(history), encoding="utf-8")
    return latest


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
    args = parser.parse_args()
    if args.command == "select":
        cases = select_cases(args.source, args.prefix, args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(cases) + "\n", encoding="utf-8")
        print(f"Selected {len(cases)} cases")
    elif args.command == "run":
        if args.timeout < 1:
            parser.error("--timeout must be positive")
        args.qpa.parent.mkdir(parents=True, exist_ok=True)
        args.qpa.unlink(missing_ok=True)
        command = [
            str(args.binary.resolve()),
            f"--deqp-caselist-file={args.caselist.resolve()}",
            f"--deqp-archive-dir={args.archive.resolve()}",
            f"--deqp-vk-library-path={args.library.resolve()}",
            f"--deqp-log-filename={args.qpa.resolve()}",
            "--deqp-log-images=disable", "--deqp-log-shader-sources=disable",
        ]
        metadata = {
            "sha": args.sha, "runUrl": args.run_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ctsRevision": args.cts_revision, "runtimeVersion": args.runtime_version,
            "baseline": args.baseline, "platform": "macOS / MoltenVK",
            "invocation": shlex.join(command), "timeoutSeconds": args.timeout,
        }
        try:
            result = subprocess.run(command, timeout=args.timeout, check=False)
            metadata.update(exitCode=result.returncode, timedOut=False)
        except subprocess.TimeoutExpired:
            print("CTS time budget exceeded; incomplete/unrun cases will be visible")
            metadata.update(exitCode=None, timedOut=True)
        args.metadata.write_text(json.dumps(metadata), encoding="utf-8")
        if not args.qpa.exists():
            raise RuntimeError("CTS produced no QPA output")
        if metadata["exitCode"] not in (0, None):
            print(f"CTS exited with {metadata['exitCode']}; recording actual QPA results")
    else:
        latest = record(args.qpa, args.caselist,
                        json.loads(args.metadata.read_text(encoding="utf-8")),
                        args.data_dir, args.history_cache)
        print(f"CTS: {latest['passed']}/{latest['total']} passed; "
              f"{latest['failed']} failed, {latest['notRun']} unrun")


if __name__ == "__main__":
    main()
