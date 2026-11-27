"""Separate train and validation processes. Only train can select a structure."""
import argparse
import hashlib
import json
import random
from fractions import Fraction
from itertools import product
from pathlib import Path
from .discovery import discover
from .expressions import evaluate, from_json
from .runtime import execute


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite_checks(rows, discovery):
    phi = from_json(discovery["selected"]["phi"])
    decoder = from_json(discovery["selected"]["decoder"])
    values = [evaluate(phi, {k: Fraction(r[k]) for k in ("d", "t")}) for r in rows]
    reflexive = [a == a for a in values]
    symmetric = [(a == b) == (b == a) for a, b in product(values, repeat=2)]
    transitive = [not (a == b and b == c) or a == c for a, b, c in product(values, repeat=3)]
    # General theorem handles arbitrary positive tau. Finite checks use only
    # observed times; final unseen task durations are not supplied to training.
    times = sorted({Fraction(r["t"]) for r in rows})
    descent = [evaluate(decoder, {"z": a, "t": t}) == evaluate(decoder, {"z": b, "t": t})
               for a, b in product(values, repeat=2) if a == b for t in times]
    return {"scope": "finite sample cross-check; not the universal proof",
            "reflexivity": {"count": len(reflexive), "passed": all(reflexive)},
            "symmetry": {"count": len(symmetric), "passed": all(symmetric)},
            "transitivity": {"count": len(transitive), "passed": all(transitive)},
            "descent": {"count": len(descent), "passed": all(descent)}}


def train(input_path, registry_path, audit_dir):
    audit_dir = Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(input_path).read_text(encoding="utf-8"))
    random.Random(1729).shuffle(rows)
    save(audit_dir / "training_received.json", rows)
    discovery = discover(rows, audit_dir / "candidates.jsonl")
    registry = {"schema": "intuition.registry.v0.1",
        "types": [{"name": "Type_1", "kind": "quotient",
                   "base_domain": {"number_system": "Q", "d": ">=0", "t": ">0"},
                   "invariant": discovery["selected"]["phi"],
                   "relation": "x~y iff invariant(x)=invariant(y)",
                   "definition": "X/~; observed instances do not exhaust X/~",
                   "representation": "induced exact rational coordinate"}],
        "instances": [{k: group[k] for k in ("id", "value")} | {"type": "Type_1"}
                      for group in discovery["groups"]],
        "operations": [{"name": "Travel", "inputs": ["Type_1", "Duration"],
                        "output": "Distance", "expression": discovery["selected"]["decoder"]}]}
    save(registry_path, registry)
    save(audit_dir / "discovery.json", discovery)
    save(audit_dir / "formal_certificate.json", discovery["proof"])
    save(audit_dir / "finite_checks.json", finite_checks(rows, discovery))
    save(audit_dir / "training_complete.json", {"registry_sha256": sha(registry_path),
        "validation_seen": False, "final_tasks_seen": False,
        "process_memory": "released when this train process exits"})


def validate(rows, registry):
    phi = from_json(registry["types"][0]["invariant"])
    instances = {Fraction(item["value"]): item["id"] for item in registry["instances"]}
    details = []
    for row in rows:
        env = {k: Fraction(row[k]) for k in ("d", "t")}
        if env["d"] < 0 or env["t"] <= 0:
            raise ValueError("Invalid validation domain")
        coordinate = evaluate(phi, env)
        known = coordinate in instances
        predicted = execute(registry, [{"instance": instances[coordinate], "duration": row["t"]}])[0]["value"] if known else None
        details.append({"id": row["id"], "d": row["d"], "t": row["t"],
                        "coordinate": str(coordinate), "known_class": known,
                        "prediction": predicted,
                        "passed": known and Fraction(predicted) == env["d"]})
    return {"passed": all(item["passed"] for item in details), "details": details,
            "selection_frozen": True, "reselection_allowed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("train", "validate"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args()
    if args.phase == "train":
        train(args.input, args.registry, args.audit)
    else:
        before = sha(args.registry)
        rows = json.loads(args.input.read_text(encoding="utf-8"))
        registry = json.loads(args.registry.read_text(encoding="utf-8"))
        result = validate(rows, registry)
        result["registry_sha256_before"] = before
        result["registry_sha256_after"] = sha(args.registry)
        save(args.audit / "validation.json", result)
        if not result["passed"]:
            raise SystemExit("Validation failed; structure was NOT reselected")


if __name__ == "__main__":
    main()
