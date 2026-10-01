#!/usr/bin/env python
"""Reproduce the reference numbers: prints published value vs NOArCO for every anchor case.

Usage:  python scripts/reproduce_reference.py [--json out.json]
Exit status 0 only if every external-benchmark case matches within its stated tolerance and no GCM
comparison fails (documented discrepancies are listed, not hidden).
"""

from __future__ import annotations

import argparse
import json
import sys

from noarco.validation import gcm_markdown, run_gcm_validation, run_validation, to_markdown


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None, help="also write the results as JSON")
    args = ap.parse_args()
    ext, gcm = run_validation(), run_gcm_validation()
    print("# Anchor cases (published values vs NOArCO)\n")
    print(to_markdown(ext))
    print("\n# Published GCM / climate-model comparisons\n")
    print(gcm_markdown(gcm))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({
                "anchor": [{"id": r.case.case_id, "source": r.case.source, "published": r.case.published,
                            "noarco": r.value, "rel_error": r.rel_error, "passed": r.passed} for r in ext],
                "gcm": [r.__dict__ for r in gcm]}, fh, indent=2)
    ok = all(r.passed for r in ext) and not any(r.status == "FAIL" for r in gcm)
    print("\nRESULT:", "all anchors reproduced" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
