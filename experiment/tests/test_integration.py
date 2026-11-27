"""Isolation, leakage, ablation, and saved-experiment acceptance checks."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from intuition import expressions
from intuition.discovery import discover
from intuition.pipeline import validate
from intuition.proof import certify
from intuition.runtime import execute
from tests.test_core import TRAIN_PAIRS, observations


ROOT = Path(__file__).resolve().parents[1]
TASKS = [
    {"instance": "v1", "duration": "7"},
    {"instance": "v2", "duration": "7"},
    {"instance": "v3", "duration": "10"},
]


def registry_from_discovery(result):
    """Test assembly follows the public registration schema, not a formula."""
    return {
        "schema": "intuition.registry.v0.1",
        "types": [{"name": "Type_1", "kind": "quotient", "invariant": result["selected"]["phi"]}],
        "instances": [
            {"id": group["id"], "type": "Type_1", "value": group["value"]}
            for group in result["groups"]
        ],
        "operations": [{
            "name": "Travel", "inputs": ["Type_1", "Duration"],
            "output": "Distance", "expression": result["selected"]["decoder"],
        }],
    }


class RuntimeIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as directory:
            result = discover(observations(TRAIN_PAIRS), Path(directory) / "candidates.jsonl")
        cls.registry = registry_from_discovery(result)

    def test_generated_registry_really_computes_unseen_tasks(self):
        results = execute(self.registry, TASKS)
        self.assertEqual([Fraction(item["value"]) for item in results], [14, 21, 25])
        self.assertTrue(all(item["type"] == "Distance" for item in results))

    def test_changed_rule_changes_results_instead_of_returning_cached_answers(self):
        changed = deepcopy(self.registry)
        changed["operations"][0]["expression"] = {"var": "z"}
        results = execute(changed, TASKS)
        self.assertEqual([Fraction(item["value"]) for item in results], [2, 3, Fraction(5, 2)])
        self.assertNotEqual(results, execute(self.registry, TASKS))

    def test_instance_type_mismatch_is_rejected(self):
        changed = deepcopy(self.registry)
        changed["instances"][0]["type"] = "Distance"
        with self.assertRaises(TypeError):
            execute(changed, TASKS[:1])

    def test_declared_operation_input_mismatch_is_rejected(self):
        changed = deepcopy(self.registry)
        changed["operations"][0]["inputs"] = ["Distance", "Duration"]
        with self.assertRaises(TypeError):
            execute(changed, TASKS[:1])

    def test_nonpositive_and_nonnumeric_durations_are_rejected(self):
        for duration in ("0", "-1", "-1/3", "NaN", "Infinity"):
            with self.subTest(duration=duration):
                with self.assertRaises(ValueError):
                    execute(self.registry, [{"instance": "v1", "duration": duration}])

    def test_unknown_instance_is_rejected(self):
        with self.assertRaises(KeyError):
            execute(self.registry, [{"instance": "missing", "duration": "7"}])

    def test_answer_fields_are_not_accepted_as_task_inputs(self):
        with self.assertRaises(ValueError):
            execute(self.registry, [{"instance": "v1", "duration": "7", "answer": "14"}])

    def test_worker_runs_from_only_its_declared_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            worker = folder / "runtime.py"
            shutil.copyfile(ROOT / "intuition" / "runtime.py", worker)
            registry_path, task_path, out_path = (folder / name for name in ("registry.json", "tasks.json", "result.json"))
            registry_path.write_text(json.dumps(self.registry), encoding="utf-8")
            task_path.write_text(json.dumps(TASKS), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(worker), "--registry", str(registry_path),
                 "--tasks", str(task_path), "--out", str(out_path)],
                cwd=folder, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual([item["value"] for item in report["results"]], ["14", "21", "25"])
        self.assertEqual(report["raw_observation_reads"], 0)
        self.assertEqual(
            {(item["file"], item["mode"]) for item in report["file_accesses"]},
            {("registry.json", "read"), ("tasks.json", "read"), ("result.json", "write")},
        )
        self.assertTrue(all(item["allowed"] for item in report["file_accesses"]))

    def test_missing_registry_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task_path, out_path = folder / "tasks.json", folder / "result.json"
            task_path.write_text(json.dumps(TASKS), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-I", "-S", str(ROOT / "intuition" / "runtime.py"),
                 "--registry", str(folder / "missing.json"), "--tasks", str(task_path), "--out", str(out_path)],
                cwd=folder, capture_output=True, text=True, timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(out_path.exists())

    def test_validation_cannot_reselect_or_mutate_the_structure(self):
        rows = observations(((10, 5), (15, 5), (25, 10)))
        frozen = deepcopy(self.registry)
        original = validate(rows, self.registry)
        self.assertTrue(original["passed"])
        rows[0]["d"] = "999"
        poisoned = validate(rows, self.registry)
        self.assertFalse(poisoned["passed"])
        self.assertTrue(poisoned["selection_frozen"])
        self.assertFalse(poisoned["reselection_allowed"])
        self.assertEqual(self.registry, frozen)


class SearchAblationTests(unittest.TestCase):
    def test_removing_division_fails_honestly_without_formula_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            try:
                with patch.object(expressions, "OPS", ("+", "-", "*")):
                    expressions.enumerate_exprs.cache_clear()
                    with self.assertRaisesRegex(ValueError, "No candidate|none"):
                        discover(observations(TRAIN_PAIRS), path)
                    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(len(records), 158)
                    self.assertTrue(all("/" not in row["expression"] for row in records))
            finally:
                expressions.enumerate_exprs.cache_clear()


class SavedExperimentTests(unittest.TestCase):
    """Recheck actual saved evidence; self-reported `passed` is insufficient."""

    @classmethod
    def setUpClass(cls):
        cls.runs_dir = Path(os.environ.get("INTUITION_RUNS", str(ROOT / "runs"))).resolve()
        path = cls.runs_dir / "reproducibility.json"
        if not path.is_file():
            raise unittest.SkipTest("Run python run_all.py first to create saved experiment evidence")
        cls.reproducibility = json.loads(path.read_text(encoding="utf-8"))
        cls.runs = cls.reproducibility["runs"]

    def load(self, run, relative):
        return json.loads((self.runs_dir / run["run"] / relative).read_text(encoding="utf-8"))

    def test_at_least_three_runs_have_identical_recomputed_artifact_hashes(self):
        self.assertGreaterEqual(len(self.runs), 3)
        paths = {
            "candidates": "audit/candidates.jsonl",
            "discovery": "audit/discovery.json",
            "formal_certificate": "audit/formal_certificate.json",
            "registry": "runtime/registry.json",
            "results": "runtime/result.json",
        }
        recomputed = []
        for run in self.runs:
            hashes = {
                key: hashlib.sha256((self.runs_dir / run["run"] / relative).read_bytes()).hexdigest()
                for key, relative in paths.items()
            }
            self.assertEqual(hashes, run["hashes"])
            recomputed.append(hashes)
        self.assertTrue(all(hashes == recomputed[0] for hashes in recomputed))
        self.assertTrue(self.reproducibility["all_equal"])

    def test_raw_input_files_are_actually_absent_in_every_run(self):
        for run in self.runs:
            folder = self.runs_dir / run["run"]
            self.assertTrue((folder / "inputs").is_dir())
            self.assertEqual(list((folder / "inputs").iterdir()), [])
            deletion = self.load(run, "audit/deletion.json")
            self.assertTrue(deletion["training_process_exited"])
            self.assertTrue(deletion["validation_process_exited"])
            self.assertTrue(deletion["input_directory_empty"])
            self.assertEqual({item["file"] for item in deletion["before"]}, {"training.json", "validation.json"})
            self.assertTrue(all(item["exists"] for item in deletion["before"]))
            for item in deletion["after"]:
                self.assertFalse(item["exists"])
                self.assertFalse((folder / "inputs" / item["file"]).exists())

    def test_logged_phase_order_puts_deletion_before_fresh_worker(self):
        expected = [
            "initial_capabilities", "training_process_exited", "structure_selected_and_registered",
            "validation_process_exited", "raw_input_files_deleted", "fresh_worker_finished",
        ]
        for run in self.runs:
            path = self.runs_dir / run["run"] / "audit" / "events.jsonl"
            events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([item["seq"] for item in events], list(range(1, 7)))
            self.assertEqual([item["stage"] for item in events], expected)
            initial = events[0]
            self.assertFalse(initial["final_tasks_available"])
            self.assertFalse(initial["validation_available"])
            self.assertFalse(initial["literal_leaves"])
            self.assertEqual(initial["max_ast_nodes"], 5)
            self.assertEqual(initial["variables"], ["d", "t"])
            self.assertEqual(set(initial["operations"]), {"+", "-", "*", "/"})
            self.assertEqual(events[-1]["result_values"], ["14", "21", "25"])

    def test_train_and_validation_records_preserve_the_registry_hash(self):
        for run in self.runs:
            registry_path = self.runs_dir / run["run"] / "runtime" / "registry.json"
            actual_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
            training = self.load(run, "audit/training_complete.json")
            validation = self.load(run, "audit/validation.json")
            self.assertFalse(training["validation_seen"])
            self.assertFalse(training["final_tasks_seen"])
            self.assertEqual(training["registry_sha256"], actual_hash)
            self.assertEqual(validation["registry_sha256_before"], actual_hash)
            self.assertEqual(validation["registry_sha256_after"], actual_hash)
            self.assertTrue(validation["selection_frozen"])
            self.assertFalse(validation["reselection_allowed"])
            self.assertEqual(len(validation["details"]), 3)
            self.assertTrue(all(item["passed"] for item in validation["details"]))

    def test_saved_workers_have_only_registered_state_and_declared_file_access(self):
        for run in self.runs:
            folder = self.runs_dir / run["run"] / "runtime"
            self.assertEqual({path.name for path in folder.iterdir()},
                             {"registry.json", "tasks.json", "runtime.py", "result.json"})
            registry = self.load(run, "runtime/registry.json")
            for instance in registry["instances"]:
                self.assertEqual(set(instance), {"id", "type", "value"})
            result = self.load(run, "runtime/result.json")
            self.assertEqual([item["value"] for item in result["results"]], ["14", "21", "25"])
            self.assertEqual(result["raw_observation_reads"], 0)
            self.assertEqual(
                {(item["file"], item["mode"]) for item in result["file_accesses"]},
                {("registry.json", "read"), ("tasks.json", "read"), ("result.json", "write")},
            )
            self.assertTrue(all(item["allowed"] for item in result["file_accesses"]))

    def test_saved_formal_certificates_can_be_rechecked_from_generated_asts(self):
        for run in self.runs:
            discovery = self.load(run, "audit/discovery.json")
            selected = discovery["selected"]
            recertified = certify(expressions.from_json(selected["phi"]), expressions.from_json(selected["decoder"]))
            self.assertEqual(self.load(run, "audit/formal_certificate.json"), recertified)
            finite = self.load(run, "audit/finite_checks.json")
            self.assertEqual(finite["reflexivity"]["count"], 6)
            self.assertEqual(finite["symmetry"]["count"], 36)
            self.assertEqual(finite["transitivity"]["count"], 216)
            self.assertTrue(all(finite[name]["passed"] for name in ("reflexivity", "symmetry", "transitivity", "descent")))


if __name__ == "__main__":
    unittest.main()
