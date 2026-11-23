"""Preserve the independent CTS dataset when either Pages publisher runs."""

import json
import urllib.error
import urllib.request
from pathlib import Path


def preserve_cts_data(pages_url: str, data_dir: Path) -> None:
    target = data_dir / "vulkan-cts"
    pending = {}
    for name in ("latest.json", "history.json", "badge.json"):
        if (target / name).exists():
            continue
        url = f"{pages_url.rstrip('/')}/data/vulkan-cts/{name}"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                pending[name] = json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            print(f"CTS dataset not yet published: {url}")
    if pending:
        target.mkdir(parents=True, exist_ok=True)
        for name, value in pending.items():
            (target / name).write_text(json.dumps(value), encoding="utf-8")
