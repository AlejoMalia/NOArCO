"""Reproducible throughput benchmark for NOArCO's analytical engines.

Measures wall-clock cost per call of representative full evaluations. It makes
no speedup claim against any external solver: compare against your own
reference solver with the same inputs if you need a ratio.

Run:  python benchmarks/benchmark_throughput.py [--n 2000] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable

from noarco.data.planets import MARS
from noarco.engines.knowledge import MechanismKnowledgeEngine
from noarco.inventory.inventory import InventoryEngine
from noarco.session import NOArCoSession


def bench(fn: Callable[[], object], n: int, repeats: int = 5) -> dict[str, float]:
    fn()  # warm-up
    per_call_us: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        per_call_us.append((time.perf_counter() - t0) / n * 1e6)
    med = statistics.median(per_call_us)
    return {"median_us": med, "min_us": min(per_call_us), "max_us": max(per_call_us),
            "calls_per_s": 1e6 / med}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    kb = MechanismKnowledgeEngine()
    session = NOArCoSession.for_mars()
    cases = {
        "InventoryEngine.evaluate (Mars, 2500 m3)": lambda: InventoryEngine.evaluate(MARS, 2500.0),
        "MechanismKnowledgeEngine.recommend_for_planet": lambda: kb.recommend_for_planet(MARS, top_n=3, verbose=False),
        "NOArCoSession.co2_feasibility": lambda: session.co2_feasibility,
        "NOArCoSession.assure (mc_samples=0)": lambda: session.assure(mc_samples=0),
    }
    results = {k: bench(fn, a.n) for k, fn in cases.items()}
    meta = {"python": sys.version.split()[0], "machine": platform.machine(),
            "system": platform.system(), "n_per_repeat": a.n, "repeats": 5}
    print(f"{'case':55s} {'median us':>10s} {'calls/s':>12s}")
    for k, r in results.items():
        print(f"{k:55s} {r['median_us']:10.1f} {r['calls_per_s']:12,.0f}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"meta": meta, "results": results}, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
