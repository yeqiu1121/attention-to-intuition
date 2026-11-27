"""Standalone execution worker: stdlib only; never imports the learner.

Example: python -I -S runtime.py --registry registry.json --tasks tasks.json
            --out result.json
Only the three declared files may be opened after installing the audit hook.
This is a cooperative I/O boundary, not an OS security sandbox.
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path


def eval_ast(node, env):
    if not isinstance(node, dict):
        raise ValueError("Invalid expression")
    if set(node) == {"var"} and node["var"] in env:
        return env[node["var"]]
    if set(node) != {"op", "args"} or node["op"] not in ("+", "-", "*", "/"):
        raise ValueError("Invalid expression fields")
    if not isinstance(node["args"], list) or len(node["args"]) != 2:
        raise ValueError("Invalid arity")
    a, b = (eval_ast(arg, env) for arg in node["args"])
    if node["op"] == "+":
        return a + b
    if node["op"] == "-":
        return a - b
    if node["op"] == "*":
        return a * b
    return a / b


def execute(registry, tasks):
    if registry["schema"] != "intuition.registry.v0.1":
        raise ValueError("Unsupported registry")
    types = {item["name"]: item for item in registry["types"]}
    instances = {item["id"]: item for item in registry["instances"]}
    operations = {item["name"]: item for item in registry["operations"]}
    operation = operations["Travel"]
    results = []
    for task in tasks:
        if set(task) != {"instance", "duration"}:
            raise ValueError("Task requires instance and duration only")
        instance = instances[task["instance"]]
        if instance["type"] not in types or operation["inputs"] != [instance["type"], "Duration"]:
            raise TypeError("Operation input type mismatch")
        duration = Fraction(task["duration"])
        if duration <= 0:
            raise ValueError("Duration must be positive")
        value = eval_ast(operation["expression"], {"z": Fraction(instance["value"]), "t": duration})
        if operation["output"] != "Distance" or value < 0:
            raise ValueError("Invalid output type or distance")
        results.append({"instance": task["instance"], "duration": str(duration),
                        "type": operation["output"], "value": str(value)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    registry_path, task_path, out_path = (p.resolve() for p in (args.registry, args.tasks, args.out))
    if len({registry_path, task_path, out_path}) != 3:
        raise ValueError("Input and output paths must be distinct")
    access_log = []

    def audit(event, arguments):
        if event == "open":
            raw_path, mode, flags = arguments
            if isinstance(raw_path, int):
                raise PermissionError("File descriptors not accepted")
            path = Path(raw_path).resolve()
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & 3 != 0
            allowed = path == out_path if writing else path in (registry_path, task_path)
            access_log.append({"file": path.name, "mode": "write" if writing else "read", "allowed": allowed})
            if not allowed:
                raise PermissionError("Execution worker attempted undeclared file access")

    sys.addaudithook(audit)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    tasks = json.loads(task_path.read_text(encoding="utf-8"))
    results = execute(registry, tasks)
    # Opening the output first includes its own write event in the audit log.
    with out_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump({"results": results, "file_accesses": access_log,
                   "worker": "fresh_process_stdlib_only", "raw_observation_reads": 0},
                  stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
