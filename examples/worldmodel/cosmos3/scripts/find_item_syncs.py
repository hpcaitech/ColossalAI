"""Find what encloses the synchronising ``_local_scalar_dense`` calls in a chrome trace.

``aten::item`` forces the device queue to drain, so one of them per few kernels turns a fully asynchronous encode into
a lock-step one. This rebuilds the CPU nesting from the trace's timestamps and prints the chain of enclosing ops for
every such call, which names the culprit without recording python stacks.
"""

import argparse
import json
from collections import Counter

TARGET = "_local_scalar_dense"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("--target", default=TARGET)
    parser.add_argument("--depth", type=int, default=5)
    args = parser.parse_args()

    with open(args.trace) as handle:
        data = json.load(handle)
    events = data["traceEvents"] if isinstance(data, dict) else data
    events = [e for e in events if e.get("ph") == "X" and e.get("dur") is not None]
    if not events:
        return
    pid = Counter(e.get("pid") for e in events).most_common(1)[0][0]
    events = [e for e in events if e.get("pid") == pid]
    events.sort(key=lambda e: (e["ts"], -e["dur"]))

    chains = Counter()
    shapes = Counter()
    parents = Counter()
    stack = []
    for event in events:
        end = event["ts"] + event["dur"]
        while stack and stack[-1][0] <= event["ts"]:
            stack.pop()
        if args.target in event.get("name", ""):
            chain = [name for _, name, _ in reversed(stack[-args.depth :])] + [event["name"]]
            chains[" <- ".join(chain)] += 1
            shapes[str(event.get("args", {}).get("Input Dims"))] += 1
            for _, _, outer in reversed(stack[-args.depth :]):
                dims = outer.get("args", {}).get("Input Dims")
                if dims:
                    parents[f"{outer['name']} {dims}"] += 1
        stack.append((end, event["name"], event))

    total = sum(chains.values())
    print(f"[{args.target}] {total} calls")
    for chain, count in chains.most_common(10):
        print(f"  {count:>5}  {chain}")
    print("[parent] enclosing op and its inputs:")
    for name, count in parents.most_common(6):
        print(f"  {count:>5}  {str(name)[:200]}")
    print(f"[{args.target}] operand shapes:")
    for shape, count in shapes.most_common(8):
        print(f"  {count:>5}  {shape[:160]}")


if __name__ == "__main__":
    main()
