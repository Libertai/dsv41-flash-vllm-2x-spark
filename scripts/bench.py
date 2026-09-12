#!/usr/bin/env python3
"""Single-stream decode throughput, by workload category.

Decode rate varies a lot by content on this model, so one number is misleading.
Counts usage.completion_tokens, never SSE chunks: with speculative decoding one chunk
covers a whole verification step, so chunk-counting under-reports by the accept length.

usage: bench.py <base_url> <model> [reps]
"""
import json, statistics, sys, time, urllib.error, urllib.request

BASE = sys.argv[1].rstrip("/"); MODEL = sys.argv[2]
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
N = 256

CASES = {
    "counting":  "Count from 1 to 300, separated by spaces.",
    "code":      "Write a Python class implementing a linked list with insert, delete and reverse. Code only.",
    "prose":     "Write a few paragraphs about the history of the Mediterranean olive trade.",
    "math":      "Compute the sum of the first 40 primes, showing each step.",
    "json":      "Emit a JSON array of 40 objects, each with id, name, and a three-word description.",
}

def guard(fn):
    try:
        return fn()
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {BASE}: {e.reason}\n"
                 "if the engine is still loading it has not bound its port yet.")

def run(prompt):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": N, "min_tokens": N,
            "ignore_eos": True, "temperature": 0}
    req = urllib.request.Request(BASE + "/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    return d["usage"]["completion_tokens"] / dt

print(f"{'category':<10} {'tok/s (median of ' + str(REPS) + ')':>24}")
allv = []
for name, prompt in CASES.items():
    vals = [guard(lambda: run(prompt)) for _ in range(REPS)]
    med = statistics.median(vals)
    allv.append(med)
    print(f"{name:<10} {med:>24.1f}   (runs: {', '.join(f'{v:.1f}' for v in vals)})")
print(f"\n{'median across categories':<34} {statistics.median(allv):.1f} tok/s")
