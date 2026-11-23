#!/usr/bin/env python3
"""Parse llvm-lit JSON results and update multi-track GitHub Pages compliance history."""

import argparse
import io
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def category_for_test(test_name: str) -> str:
    rel = test_name.split("::", 1)[-1].strip()
    parts = rel.split("/")
    if len(parts) >= 2 and parts[0] in ("Feature", "Basic", "Tools"):
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


def parse_lit_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tests = data.get("tests", [])
    counts = {"PASS": 0, "FAIL": 0, "XFAIL": 0, "XPASS": 0, "UNSUPPORTED": 0}
    categories = defaultdict(
        lambda: {"passed": 0, "failed": 0, "xfail": 0, "unsupported": 0}
    )
    failures = []
    tests_out = []

    for t in tests:
        name = t.get("name", "")
        code = t.get("code", "UNKNOWN")
        cat = category_for_test(name)

        if code in ("PASS", "FLAKYPASS"):
            counts["PASS"] += 1
            categories[cat]["passed"] += 1
            test_status = "pass"
        elif code in ("XFAIL",):
            counts["XFAIL"] += 1
            categories[cat]["xfail"] += 1
            test_status = "xfail"
        elif code in ("UNSUPPORTED",):
            counts["UNSUPPORTED"] += 1
            categories[cat]["unsupported"] += 1
            test_status = "unsupported"
        else:
            counts["FAIL"] += 1
            categories[cat]["failed"] += 1
            test_status = "fail"
            failures.append({
                "name": name.split("::", 1)[-1].strip(),
                "category": cat,
                "code": code,
                "elapsed": round(float(t.get("elapsed", 0.0)), 3),
            })
        tests_out.append({
            "name": name.split("::", 1)[-1].strip(),
            "status": test_status,
        })

    total = len(tests)
    executed = counts["PASS"] + counts["FAIL"] + counts["XFAIL"]
    pass_rate = round((100.0 * counts["PASS"] / total), 2) if total > 0 else 0.0

    cat_list = []
    for cat, c in sorted(categories.items(), key=lambda kv: (-kv[1]["failed"], kv[0])):
        cat_total = c["passed"] + c["failed"] + c["xfail"] + c["unsupported"]
        cat_rate = round((100.0 * c["passed"] / cat_total), 1) if cat_total > 0 else 0.0
        cat_list.append({
            "name": cat,
            "passed": c["passed"],
            "failed": c["failed"],
            "xfail": c["xfail"],
            "unsupported": c["unsupported"],
            "passRate": cat_rate,
        })

    return {
        "total": total,
        "executed": executed,
        "passed": counts["PASS"],
        "failed": counts["FAIL"],
        "xfail": counts["XFAIL"],
        "unsupported": counts["UNSUPPORTED"],
        "passRate": pass_rate,
        "elapsed": round(float(data.get("elapsed", 0.0)), 2),
        "categories": cat_list,
        "failures": sorted(failures, key=lambda x: x["name"]),
        "tests": sorted(tests_out, key=lambda x: x["name"]),
    }


def fetch_json(url: str):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Crucible-Compliance-CI"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"Info: no prior data fetched from {url} ({exc}).")
    return None


def fetch_json_list(url: str) -> list | None:
    loaded = fetch_json(url)
    return loaded if isinstance(loaded, list) else None


def load_existing_history(
    pages_url: str | None,
    cache_file: Path | None,
    track_id: str,
    allow_legacy_fallback: bool,
) -> list:
    if cache_file and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, list):
                    return loaded
        except Exception as exc:
            print(f"Warning: failed to read cached history {cache_file}: {exc}")

    if pages_url:
        base = pages_url.rstrip("/")
        loaded = fetch_json_list(f"{base}/data/{track_id}/history.json")
        if loaded is not None:
            return loaded
        if allow_legacy_fallback:
            loaded = fetch_json_list(f"{base}/data/history.json")
            if loaded is not None:
                return loaded

    return []


def load_track_latest(
    pages_url: str | None, track_id: str, history: list
) -> dict | None:
    if pages_url:
        latest = fetch_json(
            f"{pages_url.rstrip('/')}/data/{track_id}/latest.json"
        )
        if isinstance(latest, dict):
            return latest
    if history:
        return {**history[-1], "failures": []}
    return None


def parse_track_spec(spec: str) -> tuple[str, str, Path]:
    parts = spec.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"invalid --track-lit '{spec}' (expected id=label=path)"
        )
    return parts[0].strip(), parts[1].strip(), Path(parts[2].strip())


def github_api_json(url: str) -> dict | None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "Crucible-Compliance-CI",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Warning: external GitHub data unavailable from {url}: {exc}")
        return None


def normalize_external_tests(data: dict) -> list:
    tests = []
    for test in data.get("tests", []):
        name = str(test.get("name", "")).split("::", 1)[-1].strip()
        code = str(test.get("code", "UNKNOWN"))
        if code in ("PASS", "FLAKYPASS"):
            status = "pass"
        elif code == "XFAIL":
            status = "xfail"
        elif code == "UNSUPPORTED":
            status = "unsupported"
        else:
            status = "fail"
        if name:
            tests.append({"name": name, "status": status})
    return sorted(tests, key=lambda test: test["name"])


def fetch_external_comparisons() -> dict:
    summary_url = (
        "https://raw.githubusercontent.com/farzonl/"
        "offload-test-suite-badges/badges/summary.json"
    )
    summary = fetch_json(summary_url)
    workflows = summary.get("workflows", {}) if isinstance(summary, dict) else {}
    comparisons = {
        "generatedAt": summary.get("generatedAt", "") if isinstance(summary, dict) else "",
        "repository": "llvm/offload-test-suite",
        "runs": [],
    }

    preferred_keys = (
        "windows-intel-dxc-vk",
        "windows-amd-dxc-lavapipe",
        "windows-intel-clang-vk",
        "windows-amd-clang-lavapipe",
        "macos-dxc-mtl",
        "macos-clang-mtl",
        "windows-intel-dxc-d3d12",
        "windows-nvidia-dxc-d3d12",
        "windows-amd-dxc-d3d12",
        "windows-intel-clang-d3d12",
        "windows-nvidia-clang-d3d12",
        "windows-amd-clang-d3d12",
    )
    for key in preferred_keys:
        workflow = workflows.get(key)
        if not isinstance(workflow, dict):
            continue
        run_id = workflow.get("runId")
        tests = []
        if run_id:
            artifacts_url = (
                f"https://api.github.com/repos/llvm/offload-test-suite/"
                f"actions/runs/{run_id}/artifacts?per_page=100"
            )
            artifact_data = github_api_json(artifacts_url)
            if artifact_data:
                token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
                headers = {"User-Agent": "Crucible-Compliance-CI"}
                if token:
                    headers["Authorization"] = f"Bearer {token}"
                for artifact in artifact_data.get("artifacts", []):
                    name = artifact.get("name", "")
                    if artifact.get("expired") or not name.startswith("test-results-"):
                        continue
                    dl_url = artifact.get("archive_download_url")
                    if not dl_url:
                        continue
                    try:
                        req = urllib.request.Request(dl_url, headers=headers)
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            raw_zip = resp.read()
                        with zipfile.ZipFile(io.BytesIO(raw_zip)) as zf:
                            for member in zf.namelist():
                                if not member.endswith("testresults.json"):
                                    continue
                                lit_data = json.loads(
                                    zf.read(member).decode("utf-8")
                                )
                                candidate = normalize_external_tests(lit_data)
                                if len(candidate) > len(tests):
                                    tests = candidate
                    except (urllib.error.HTTPError, urllib.error.URLError,
                            TimeoutError, zipfile.BadZipFile, json.JSONDecodeError,
                            UnicodeDecodeError):
                        continue

        platform = "vulkan" if "-vk" in key or "-lavapipe" in key else (
            "metal" if "-mtl" in key else "directx"
        )
        if "-dxc-" in key:
            compiler = "DXC"
        elif "-clang-" in key:
            compiler = "Clang"
        else:
            compiler = workflow.get("label", "")
        label = f"{workflow.get('label', compiler)} · {key.replace('-', ' ').title()}"
        comparisons["runs"].append({
            "id": key,
            "workflow": workflow.get("workflow", f"{key}.yaml"),
            "label": label,
            "platform": platform,
            "compiler": compiler,
            "runId": run_id,
            "runUrl": workflow.get("runUrl", ""),
            "updatedAt": workflow.get("updatedAt", ""),
            "results": workflow.get("results", {}),
            "tests": tests,
        })
    return comparisons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lit-json", type=Path, default=None,
                        help="Legacy single-track lit JSON (recorded as dxil-to-metal)")
    parser.add_argument("--track-lit", action="append", default=[],
                        help="Multi-track input: <track_id>=<label>=<lit_json_path>")
    parser.add_argument("--site-src", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--message", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--pages-url", default="")
    parser.add_argument("--history-cache", type=Path, default=None,
                        help="Legacy single-track history cache file")
    parser.add_argument("--history-cache-dir", type=Path, default=None,
                        help="Directory for per-track <track_id>.json history caches")
    args = parser.parse_args()

    tracks_in: list[tuple[str, str, Path]] = []
    for raw in args.track_lit:
        tracks_in.append(parse_track_spec(raw))
    if args.lit_json:
        tracks_in.append(("dxil-to-metal", "DXIL -> Metal (macOS)", args.lit_json))
    if not tracks_in:
        parser.error("at least one of --lit-json or --track-lit is required")

    ts = args.timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for item in args.site_src.iterdir():
        dst = args.out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    data_dir = args.out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    ignored_track_ids = {"history", "latest", "badge", "tracks"}
    previous_tracks = {}
    if args.pages_url:
        loaded_tracks = fetch_json(
            f"{args.pages_url.rstrip('/')}/data/tracks.json"
        )
        if isinstance(loaded_tracks, list):
            previous_tracks = {
                track.get("id"): track
                for track in loaded_tracks
                if isinstance(track, dict)
                and track.get("id")
                and track.get("id") not in ignored_track_ids
            }

    tracks_manifest = []
    processed_tracks = set()
    for idx, (track_id, label, lit_path) in enumerate(tracks_in):
        processed_tracks.add(track_id)
        cache_file = None
        if args.history_cache_dir:
            cache_file = args.history_cache_dir / f"{track_id}.json"
        elif args.history_cache and (track_id == "dxil-to-metal" or idx == 0):
            cache_file = args.history_cache

        history = load_existing_history(
            args.pages_url,
            cache_file,
            track_id,
            allow_legacy_fallback=(track_id == "dxil-to-metal"),
        )
        did_run = lit_path.exists()
        if not did_run:
            print(f"Warning: preserving prior track '{track_id}'; missing {lit_path}")
            entry = load_track_latest(args.pages_url, track_id, history)
            if not entry:
                continue
        else:
            summary = parse_lit_json(lit_path)
            entry = {
                "sha": args.sha,
                "shortSha": args.sha[:7],
                "message": args.message.splitlines()[0][:120] if args.message else "",
                "author": args.author,
                "timestamp": ts,
                "runUrl": args.run_url,
                "track": track_id,
                "trackLabel": label,
                "total": summary["total"],
                "executed": summary["executed"],
                "passed": summary["passed"],
                "failed": summary["failed"],
                "xfail": summary["xfail"],
                "unsupported": summary["unsupported"],
                "passRate": summary["passRate"],
                "elapsed": summary["elapsed"],
                "categories": summary["categories"],
                "failures": summary["failures"],
                "tests": summary["tests"],
            }
            history = [h for h in history if h.get("sha") != args.sha]
            history.append(entry)
            history.sort(key=lambda h: h.get("timestamp", ""))
            entry = {**entry, "failures": summary["failures"], "tests": summary["tests"]}

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        track_dir = data_dir / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        with open(track_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(track_dir / "latest.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        badge_obj = {
            "schemaVersion": 1,
            "label": label,
            "message": f"{entry['passed']} Passed ({entry['passRate']}%)",
            "color": "brightgreen" if entry["passRate"] >= 85.0 else "green",
        }
        with open(track_dir / "badge.json", "w", encoding="utf-8") as f:
            json.dump(badge_obj, f, indent=2)

        if track_id == "dxil-to-metal" or (
            idx == 0 and not (data_dir / "history.json").exists()
        ):
            with open(data_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            with open(data_dir / "latest.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2)
            with open(data_dir / "badge.json", "w", encoding="utf-8") as f:
                json.dump(badge_obj, f, indent=2)

        tracks_manifest.append({
            "id": track_id,
            "label": label,
            "passed": entry["passed"],
            "total": entry["total"],
            "passRate": entry["passRate"],
            "historyUrl": f"data/{track_id}/history.json",
            "latestUrl": f"data/{track_id}/latest.json",
        })
        if did_run:
            print(
                f"[{track_id}] Recorded {entry['shortSha']}: "
                f"{entry['passed']}/{entry['total']} passed ({entry['passRate']}%, "
                f"{entry['executed']} executed), {entry['failed']} failed, "
                f"{entry['xfail']} xfail, {entry['unsupported']} unsupported. "
                f"History length: {len(history)}"
            )

    for track_id, track in previous_tracks.items():
        if track_id in processed_tracks:
            continue
        cache_file = (
            args.history_cache_dir / f"{track_id}.json"
            if args.history_cache_dir
            else None
        )
        history = load_existing_history(
            args.pages_url, cache_file, track_id, allow_legacy_fallback=False
        )
        entry = load_track_latest(args.pages_url, track_id, history)
        if not history or not entry:
            continue
        track_dir = data_dir / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        with open(track_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(track_dir / "latest.json", "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
        badge_obj = {
            "schemaVersion": 1,
            "label": track.get("label", track_id),
            "message": f"{entry['passed']} Passed ({entry['passRate']}%)",
            "color": "brightgreen" if entry["passRate"] >= 85.0 else "green",
        }
        with open(track_dir / "badge.json", "w", encoding="utf-8") as f:
            json.dump(badge_obj, f, indent=2)
        tracks_manifest.append(track)

    with open(data_dir / "tracks.json", "w", encoding="utf-8") as f:
        json.dump(tracks_manifest, f, indent=2)

    comparisons = fetch_external_comparisons()
    with open(data_dir / "comparisons.json", "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)

    if args.pages_url:
        for fname in ("history.json", "latest.json", "badge.json"):
            if (data_dir / fname).exists():
                continue
            previous = fetch_json(
                f"{args.pages_url.rstrip('/')}/data/{fname}"
            )
            if previous is not None:
                with open(data_dir / fname, "w", encoding="utf-8") as f:
                    json.dump(previous, f, indent=2)

    # Preserve benchmark histories if already deployed on Pages
    if args.pages_url:
        for fname in ("bench-history.json", "bench-history-macos.json"):
            if (data_dir / fname).exists():
                continue
            loaded = fetch_json_list(args.pages_url.rstrip("/") + f"/data/{fname}")
            if loaded is not None:
                with open(data_dir / fname, "w", encoding="utf-8") as f:
                    json.dump(loaded, f, indent=2)


if __name__ == "__main__":
    main()
