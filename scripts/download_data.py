"""Download the two datasets used in the CAPRF++ paper.

Dataset 1 - GWA-T-12 Bitbrains (fastStorage, 1,250 VMs, 5-minute samples, ~1 month).
    Official host http://gwa.ewi.tudelft.nl is offline, so we pull the raw, unmodified
    per-VM CSVs from a public research mirror on GitHub
    (muse-research-lab/cloud-forecast-data-persistence, folder bitbrains/1201308 = fastStorage).

Dataset 2 - Kaggle "Cloud Task Scheduling Dataset" (ziya07/cloud-task-scheduling-dataset).
    Kaggle needs a login, so by default we pull an identical public copy (20,000 rows, same
    columns as the paper describes). If you have Kaggle credentials you can instead run:
        kaggle datasets download ziya07/cloud-task-scheduling-dataset -p data/raw/kaggle --unzip

Usage:
    python scripts/download_data.py               # 300 Bitbrains VMs + Kaggle CSV
    python scripts/download_data.py --vms 1250    # the full fastStorage trace (~1.1 GB)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BB_REPO = "muse-research-lab/cloud-forecast-data-persistence"
BB_DIR = "bitbrains/1201308"
BB_RAW = f"https://raw.githubusercontent.com/{BB_REPO}/HEAD/{BB_DIR}/{{name}}"
KAGGLE_MIRRORS = [
    "https://raw.githubusercontent.com/mayura-andrew/cloudpulse-sim/HEAD/cloud_task_scheduling_dataset_20k.csv",
    "https://raw.githubusercontent.com/Mohit-Talgotra/AIScheduler/HEAD/src/data/dataset.csv",
]
KAGGLE_HEADER = ("Task_ID,CPU_Usage (%),RAM_Usage (MB),Disk_IO (MB/s),Network_IO (MB/s),"
                 "Priority,VM_ID,Execution_Time (s),Target (Optimal Scheduling)")


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception:  # corporate proxies often break Python's TLS store; curl uses the system one
        subprocess.run(["curl", "-sSfL", "-m", "120", "-o", str(tmp), url], check=True)
    tmp.rename(dest)


def bitbrains_file_list() -> list[str]:
    api = f"https://api.github.com/repos/{BB_REPO}/git/trees/HEAD?recursive=1"
    try:
        with urllib.request.urlopen(api, timeout=60) as r:
            tree = json.load(r)["tree"]
    except Exception:
        out = subprocess.run(["curl", "-sSfL", "-m", "120", api], check=True, capture_output=True).stdout
        tree = json.loads(out)["tree"]
    names = [t["path"].rsplit("/", 1)[1] for t in tree
             if t["path"].startswith(BB_DIR + "/") and t["path"].endswith(".csv")]
    return sorted(names, key=lambda n: int(n.split(".")[0]))


def download_bitbrains(n_vms: int) -> None:
    out = RAW / "bitbrains" / "fastStorage"
    names = bitbrains_file_list()[:n_vms]
    todo = [n for n in names if not (out / n).exists()]
    print(f"[bitbrains] {len(names)} VMs requested, {len(todo)} to download -> {out}")

    def one(n: str) -> None:
        fetch(BB_RAW.format(name=n), out / n)

    with ThreadPoolExecutor(16) as ex:
        for i, _ in enumerate(ex.map(one, todo), 1):
            if i % 50 == 0:
                print(f"  {i}/{len(todo)}")
    print("[bitbrains] done")


def download_kaggle() -> None:
    dest = RAW / "kaggle" / "cloud_task_scheduling_dataset.csv"
    if dest.exists():
        print(f"[kaggle] already present: {dest}")
        return
    for url in KAGGLE_MIRRORS:
        try:
            fetch(url, dest)
            header = dest.read_text().splitlines()[0].strip()
            if header.replace("_", "").lower().startswith("taskid"):
                if header != KAGGLE_HEADER:  # second mirror strips the units; restore canonical names
                    lines = dest.read_text().splitlines()
                    lines[0] = KAGGLE_HEADER
                    dest.write_text("\n".join(lines) + "\n")
                print(f"[kaggle] saved {dest} from {url}")
                return
        except Exception as e:  # try the next mirror
            print(f"[kaggle] mirror failed ({url}): {e}")
    sys.exit("Could not download the Kaggle dataset; download it manually from "
             "https://www.kaggle.com/datasets/ziya07/cloud-task-scheduling-dataset into data/raw/kaggle/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vms", type=int, default=300, help="number of Bitbrains VMs to fetch (max 1250)")
    args = ap.parse_args()
    download_kaggle()
    download_bitbrains(args.vms)
