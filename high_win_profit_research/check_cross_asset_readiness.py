import csv
import re
from collections import defaultdict
from pathlib import Path


RESEARCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RESEARCH_DIR.parent
DATA_DIRS = [PROJECT_ROOT, PROJECT_ROOT / "數據集"]
OUTPUT_FILE = RESEARCH_DIR / "cross_asset_readiness.csv"

REQUIRED_TIMEFRAMES = {
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M10",
    "M12",
    "M15",
    "M20",
    "M30",
    "H1",
    "H2",
    "H3",
    "H4",
    "H6",
    "H8",
    "H12",
    "Daily",
    "Weekly",
    "Monthly",
}

FILENAME_RE = re.compile(
    r"^(?P<symbol>.+)_(?P<timeframe>M\d+|H\d+|Daily|Weekly|Monthly)_"
    r"(?P<start>\d{12})_(?P<end>\d{12})\.csv$",
    re.IGNORECASE,
)


def scan_assets():
    assets = defaultdict(dict)
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            continue
        for path in data_dir.glob("*.csv"):
            match = FILENAME_RE.match(path.name)
            if not match:
                continue
            symbol = match.group("symbol")
            timeframe = match.group("timeframe")
            assets[symbol][timeframe] = path
    return assets


def main():
    assets = scan_assets()
    rows = []
    for symbol, timeframe_paths in sorted(assets.items()):
        available = set(timeframe_paths)
        missing = sorted(REQUIRED_TIMEFRAMES - available)
        rows.append(
            {
                "symbol": symbol,
                "available_timeframes": len(available),
                "required_timeframes": len(REQUIRED_TIMEFRAMES),
                "ready_for_current_pipeline": not missing,
                "missing_timeframes": ",".join(missing),
            }
        )

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "symbol",
                "available_timeframes",
                "required_timeframes",
                "ready_for_current_pipeline",
                "missing_timeframes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Found {len(rows)} asset dataset(s).")
    for row in rows:
        status = "ready" if row["ready_for_current_pipeline"] else "missing data"
        print(
            f"{row['symbol']}: {status} "
            f"({row['available_timeframes']}/{row['required_timeframes']} timeframes)"
        )
        if row["missing_timeframes"]:
            print(f"  missing: {row['missing_timeframes']}")
    print(f"Wrote {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
