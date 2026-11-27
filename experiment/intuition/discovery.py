"""Exhaustive unsupervised grouping plus supervised distance reconstruction.

Only the numeric observations enter scoring. IDs are used for audit output.
No hidden labels, validation data, final tasks, or expected answers enter here.
"""
import json
from fractions import Fraction
from pathlib import Path
from .expressions import enumerate_exprs, evaluate, render, size, to_json
from .proof import certify


def discover(rows, trace_path):
    if len(rows) < 4:
        raise ValueError("Need repeated observations of multiple classes")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Observation IDs must be unique")
    envs = [{k: Fraction(row[k]) for k in ("d", "t")} for row in rows]
    if any(e["d"] < 0 or e["t"] <= 0 for e in envs):
        raise ValueError("Domain is rational d>=0, t>0")
    if len({(e["d"], e["t"]) for e in envs}) != len(envs):
        raise ValueError("Duplicate observations cannot evidence compression")
    candidates = enumerate_exprs(("d", "t"))
    decoders = enumerate_exprs(("z", "t"))
    eligible = []
    records = []
    for index, phi in enumerate(candidates):
        record = {"index": index, "phi": to_json(phi), "expression": render(phi),
                  "nodes": size(phi), "decoder_scores": []}
        try:
            values = [evaluate(phi, env) for env in envs]
        except ZeroDivisionError:
            record["rejected"] = "undefined_on_training"
            records.append(record)
            continue
        groups = {}
        for row, value in zip(rows, values):
            groups.setdefault(value, []).append(row["id"])
        record["values"] = [str(v) for v in values]
        record["groups"] = [{"value": str(v), "members": sorted(members)}
                            for v, members in sorted(groups.items())]
        if len(groups) == 1:
            record["rejected"] = "constant_on_training"
        elif len(groups) == len(rows):
            record["rejected"] = "no_compression"
        elif min(map(len, groups.values())) < 2:
            record["rejected"] = "singleton_class"
        else:
            for decoder in decoders:
                matches, undefined = 0, 0
                for env, value in zip(envs, values):
                    try:
                        matches += evaluate(decoder, {"z": value, "t": env["t"]}) == env["d"]
                    except ZeroDivisionError:
                        undefined += 1
                score = {"expression": render(decoder), "nodes": size(decoder),
                         "exact_matches": matches, "undefined": undefined,
                         "residual_count": len(rows) - matches}
                record["decoder_scores"].append(score)
                if matches == len(rows) and not undefined:
                    eligible.append((size(phi) + size(decoder), render(phi), render(decoder),
                                     phi, decoder, groups))
            if not any(s["exact_matches"] == len(rows) for s in record["decoder_scores"]):
                record["rejected"] = "no_exact_decoder"
            else:
                record["rejected"] = None
        records.append(record)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    if not eligible:
        raise ValueError("No candidate survives: discovery failed within this search bound")
    eligible.sort(key=lambda item: item[:3])
    min_score = eligible[0][0]
    top_ties = [{"phi": item[1], "decoder": item[2], "description_length": item[0]}
                for item in eligible if item[0] == min_score]
    proof_attempts = []
    # Totality and exact symbolic reconstruction are mandatory gates, not
    # target-specific templates. Failure to certify is explicitly recorded.
    for length, _, _, phi, decoder, groups in eligible:
        try:
            proof = certify(phi, decoder)
        except ValueError as exc:
            proof_attempts.append({"phi": render(phi), "decoder": render(decoder),
                                   "status": "uncertified", "reason": str(exc)})
            continue
        selected = {"phi": to_json(phi), "decoder": to_json(decoder),
                    "phi_text": render(phi), "decoder_text": render(decoder),
                    "description_length": length}
        proof_attempts.append({"phi": render(phi), "decoder": render(decoder), "status": "certified"})
        break
    else:
        raise ValueError("Empirical fits found but none has a domain certificate")
    # Tie diagnostics certify all minimum-length empirical fits, including
    # alternatives not traversed before the first winner.
    tie_certificates = []
    for item in eligible:
        if item[0] != min_score:
            break
        try:
            certify(item[3], item[4])
            status, reason = "certified", None
        except ValueError as exc:
            status, reason = "uncertified", str(exc)
        tie_certificates.append({"phi": item[1], "decoder": item[2], "status": status, "reason": reason})
    # Canonical public names are assigned by the smallest observed (d,t) in
    # each class. This uses no hidden state labels and survives input shuffling.
    by_id = {row["id"]: env for row, env in zip(rows, envs)}
    ordered = sorted(groups.items(), key=lambda item: min(
        (by_id[i]["d"], by_id[i]["t"]) for i in item[1]))
    classes = [{"id": f"v{i}", "value": str(value), "members": sorted(members)}
               for i, (value, members) in enumerate(ordered, 1)]
    histogram = {}
    for record in records:
        reason = record["rejected"] or "has_exact_decoder"
        histogram[reason] = histogram.get(reason, 0) + 1
    return {"selected": selected, "groups": classes, "proof": proof,
            "candidate_count": len(candidates), "decoder_count": len(decoders),
            "eligible_pair_count": len(eligible), "filter_counts": histogram,
            "top_ties": top_ties, "tie_certificates": tie_certificates,
            "proof_attempts": proof_attempts,
            "selection_rule": ["defined on all training observations", "nonconstant on training",
                "at least two classes, every class has at least two distinct observations",
                "zero exact reconstruction residual", "minimum L(phi)+L(g), one token per AST node",
                "lexicographic rendered phi, then g; no physical meaning in tie-break",
                "mandatory generic full-domain totality and reconstruction certificate"]}
