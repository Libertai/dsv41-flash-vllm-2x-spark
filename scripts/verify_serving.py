#!/usr/bin/env python3
"""Verify a DeepSeek-V4.1-Flash endpoint is actually CORRECT before you benchmark it.

Two faults in this stack produce a server that starts, answers, and benchmarks normally
while being completely wrong, so none of these checks are optional:

  1. logprobs   -- NaN output is invisible in `content`; it surfaces here as an HTTP 400
                   "Out of range float values are not JSON compliant: nan".
  2. known answer -- catches garbage that happens to decode to printable text.
  3. DEFAULT request shape, no optional flags -- thinking is ON by default, so a probe
                   that always sends {"thinking": false} never tests the real default.
  4. tool round trip.

usage: verify_serving.py <base_url> <model> [--expect-reasoning]
"""
import json, sys, urllib.error, urllib.request

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
fails = []

def post(path, body, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=timeout)), None
    except urllib.error.HTTPError as e:
        return None, e.read().decode()[:300]

def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)

print("1. logprobs (NaN detector)")
d, err = post("/completions", {"model": MODEL, "prompt": "The capital of France is",
                               "max_tokens": 5, "temperature": 0, "logprobs": 5})
if err and "nan" in err.lower():
    check("logprobs finite", False, "MODEL IS EMITTING NaN -- see Troubleshooting; do NOT trust any benchmark")
elif err:
    check("logprobs finite", False, err)
else:
    lp = (d["choices"][0].get("logprobs") or {}).get("token_logprobs") or []
    check("logprobs finite", bool(lp) and all(x is None or x == x for x in lp), f"{lp[:3]}")

print("2. known answer")
d, err = post("/chat/completions", {"model": MODEL, "max_tokens": 24, "temperature": 0,
              "chat_template_kwargs": {"thinking": False},
              "messages": [{"role": "user", "content": "What is 17*19? Answer with the number only."}]})
txt = "" if err else (d["choices"][0]["message"].get("content") or "")
check("17*19 == 323", "323" in txt, repr(txt[:80]) if not err else err)

print("3. default request shape (no optional flags)")
d, err = post("/chat/completions", {"model": MODEL, "max_tokens": 260, "temperature": 0,
              "messages": [{"role": "user", "content": "What is 17*19?"}]})
if err:
    check("default shape", False, err)
else:
    m = d["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning") or m.get("reasoning_content")
    check("no raw </think> in content", "</think>" not in content and "<think>" not in content,
          repr(content[:90]))
    check("answer present in content", "323" in content, repr(content[:90]))
    if reasoning:
        print(f"         reasoning separated ({len(reasoning)} chars) -- parser is wired")
    else:
        print("         note: no reasoning field; fine only if the server defaults thinking OFF")

print("4. tool call round trip")
tools = [{"type": "function", "function": {"name": "get_weather",
          "description": "Get current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                         "required": ["city"]}}}]
d, err = post("/chat/completions", {"model": MODEL, "max_tokens": 300, "temperature": 0,
              "messages": [{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
              "tools": tools})
if err:
    check("tool call emitted", False, err)
else:
    tc = d["choices"][0]["message"].get("tool_calls") or []
    check("tool call emitted", bool(tc) and tc[0]["function"]["name"] == "get_weather",
          (tc[0]["function"]["arguments"] if tc else "none"))

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {', '.join(fails)}")
    sys.exit(1)
print("All checks passed -- safe to benchmark.")
