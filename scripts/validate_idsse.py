"""Write the IDSSE source-quality manifest for the local XML directory."""
import argparse
import json
from pathlib import Path

from soccer_tactics.analytics.idsse_validation import validate_idsse_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate_idsse_dataset(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    accepted = sum(row["status"] == "accepted" for row in report["candidates"])
    print(f"IDSSE audit: {accepted}/{report['candidate_count']} candidates accepted; "
          f"manifest: {args.output}")
    return 0 if report["candidate_count"] == 7 and not report["errors"] and accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
