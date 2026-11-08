"""Download the MSU/ORNL power system attack datasets (Adhikari, Pan, Morris et al.)
into data/raw/ and extract them.
"""
from pathlib import Path

import py7zr
import requests

BASE = "http://www.ece.uah.edu/~thm0009/icsdatasets/"
FILES = ["PowerSystem_Dataset_README.pdf", "multiclass.7z", "triple.7z", "binaryAllNaturalPlusNormalVsAttacks.7z"]
RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        out = RAW / f
        if not out.exists():
            print("[get ]", f)
            r = requests.get(BASE + f, timeout=300)
            r.raise_for_status()
            out.write_bytes(r.content)
        if out.suffix == ".7z":
            dest = RAW / out.stem
            if not dest.exists():
                with py7zr.SevenZipFile(out) as z:
                    z.extractall(dest)


if __name__ == "__main__":
    main()
