#!/usr/bin/env python
"""Verify a NOArCO result record: schema check + deterministic replay from its own inputs.

Usage:  python scripts/verify_record.py record.json
Exit 0: valid schema and the replay reproduces input hash, verdict, requirements and headroom.
Exit 1: invalid schema or the replay differs.  Exit 2: unreadable file.
A record made by another version/constants/code is still replayed; ``same_environment`` says whether it was
produced by exactly this code.
"""

from __future__ import annotations

import json
import sys

from noarco.verdict import replay, validate_record


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        with open(argv[1]) as fh:
            record = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"cannot read {argv[1]}: {exc}")
        return 2
    problems = validate_record(record)
    if problems:
        print("INVALID:", *problems, sep="\n  - ")
        return 1
    r = replay(record)
    print(f"schema: valid | replay: {'MATCH' if r.matches else 'DIFFERS ' + str(r.differences)} | "
          f"same environment (version, constants, code): {r.same_environment}")
    return 0 if r.matches else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
