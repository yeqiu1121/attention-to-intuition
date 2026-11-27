"""Objective regression checks, including known failure modes.

The expected grouping and results in these tests are evaluator knowledge.  They
are deliberately outside the learner and never supplied to expression search.
Run from the project directory with ``python -m unittest discover -v``.
"""

from collections import Counter
from fractions import Fraction
from pathlib import Path
import json
import tempfile
import unittest

from intuition.expressions import (
    Expr, enumerate_exprs, evaluate, from_json, render, size, to_json,
)
from intuition.proof import certify
from intuition.discovery import discover


def leaf(name):
    return Expr(name)


def binary(op, left, right):
    return Expr(op, (left, right))


def observations(pairs):
    return [
        {"id": "o" + str(index + 1), "d": str(d), "t": str(t)}
        for index, (d, t) in enumerate(pairs)
    ]


TRAIN_PAIRS = ((2, 1), (8, 4), (3, 1), (12, 4), (5, 2), (20, 8))


class GrammarTests(unittest.TestCase):
    def test_complete_bounded_grammar_has_274_ordered_asts(self):
        expressions = enumerate_exprs(("d", "t"), max_nodes=5)
        self.assertEqual(len(expressions), 274)
        self.assertEqual(Counter(size(e) for e in expressions), {1: 2, 3: 16, 5: 256})
        self.assertEqual(len({render(e) for e in expressions}), 274)
        self.assertTrue(all(size(e) <= 5 for e in expressions))

    def test_grammar_has_only_variables_and_four_binary_operations(self):
        def inspect(expr):
            if not expr.args:
                self.assertIn(expr.kind, {"d", "t"})
            else:
                self.assertIn(expr.kind, {"+", "-", "*", "/"})
                self.assertEqual(len(expr.args), 2)
                for child in expr.args:
                    inspect(child)

        for expr in enumerate_exprs(("d", "t"), max_nodes=5):
            inspect(expr)

    def test_search_is_parameterized_by_variable_names(self):
        expressions = enumerate_exprs(("z", "t"), max_nodes=5)
        self.assertEqual(len(expressions), 274)
        self.assertIn(binary("/", leaf("t"), leaf("z")), expressions)
        self.assertIn(binary("*", leaf("z"), leaf("t")), expressions)

    def test_exact_fraction_arithmetic(self):
        expression = binary("/", leaf("d"), leaf("t"))
        actual = evaluate(expression, {"d": Fraction(1, 3), "t": Fraction(7, 11)})
        self.assertIsInstance(actual, Fraction)
        self.assertEqual(actual, Fraction(11, 21))

    def test_division_by_zero_is_rejected(self):
        expression = binary("/", leaf("d"), binary("-", leaf("t"), leaf("t")))
        with self.assertRaises((ZeroDivisionError, ValueError)):
            evaluate(expression, {"d": Fraction(2), "t": Fraction(1)})

    def test_every_ast_round_trips_without_execution(self):
        for expression in enumerate_exprs(("d", "t"), max_nodes=5):
            self.assertEqual(from_json(to_json(expression)), expression)

    def test_malformed_serialized_asts_are_rejected(self):
        for payload in (
            None,
            1,
            "d/t",
            {},
            {"var": "d", "op": "+"},
            {"op": "__import__", "args": []},
            {"op": "+", "args": [{"var": "d"}]},
            {"op": "/", "args": [{"var": "d"}, {"var": "t"}], "code": "answer"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises((ValueError, TypeError, KeyError)):
                    from_json(payload)


class FormalCertificateTests(unittest.TestCase):
    def setUp(self):
        self.d, self.t, self.z = leaf("d"), leaf("t"), leaf("z")

    def test_total_rational_reconstruction_is_certified(self):
        certificate = certify(binary("/", self.d, self.t), binary("*", self.z, self.t))
        self.assertIsInstance(certificate, dict)
        self.assertTrue(certificate)

    def test_wrong_decoder_is_not_certified(self):
        with self.assertRaises(ValueError):
            certify(binary("/", self.d, self.t), binary("+", self.z, self.t))

    def test_cancellation_does_not_erase_a_singularity(self):
        # (d*d)/d simplifies to d algebraically, but the original AST fails at d=0.
        singular = binary("/", binary("*", self.d, self.d), self.d)
        with self.assertRaises(ValueError):
            certify(singular, self.z)

    def test_reciprocal_representation_is_not_total_at_zero_distance(self):
        # It fits the six positive-distance rows, yet cannot represent d=0.
        with self.assertRaises(ValueError):
            certify(binary("/", self.t, self.d), binary("/", self.t, self.z))

    def test_identically_zero_denominator_is_rejected(self):
        invalid = binary("/", self.d, binary("-", self.t, self.t))
        with self.assertRaises(ValueError):
            certify(invalid, binary("*", self.z, self.t))

    def test_future_operation_must_preserve_nonnegative_distance(self):
        # This reconstructs observed d exactly, but can go negative at a new time.
        with self.assertRaises(ValueError):
            certify(binary("+", self.d, self.t), binary("-", self.z, self.t))

    def test_checker_accepts_a_different_total_nonnegative_rational_program(self):
        # Generic proof checking must not be a recognition template for d/t.
        certificate = certify(binary("*", self.d, self.t), binary("/", self.z, self.t))
        self.assertTrue(certificate["checks"]["output_nonnegative"])


class DiscoveryTests(unittest.TestCase):
    def run_discovery(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            trace_path = Path(directory) / "candidates.jsonl"
            result = discover(rows, trace_path)
            self.assertTrue(trace_path.is_file(), "Every discovery must leave a candidate trace")
            self.assertGreater(trace_path.stat().st_size, 0)
            return result

    def selected_pair(self, result):
        selected = result["selected"]
        return from_json(selected["phi"]), from_json(selected["decoder"])

    def assert_reconstructs_and_groups(self, rows, result):
        phi, decoder = self.selected_pair(result)
        values = []
        for row in rows:
            d, t = Fraction(row["d"]), Fraction(row["t"])
            value = evaluate(phi, {"d": d, "t": t})
            values.append(value)
            self.assertEqual(evaluate(decoder, {"z": value, "t": t}), d)
        self.assertGreater(len(set(values)), 1, "A constant must never win")
        partition = {frozenset(i for i, value in enumerate(values) if value == x) for x in values}
        self.assertEqual(partition, {frozenset((0, 1)), frozenset((2, 3)), frozenset((4, 5))})

    def test_discovers_nontrivial_groups_and_exact_decoder(self):
        rows = observations(TRAIN_PAIRS)
        self.assert_reconstructs_and_groups(rows, self.run_discovery(rows))

    def test_candidate_trace_includes_rejected_constants_and_every_ast(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.jsonl"
            discover(observations(TRAIN_PAIRS), path)
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(records), 274)
        by_expression = {record["expression"]: record for record in records}
        self.assertEqual(by_expression["(d/d)"]["rejected"], "constant_on_training")
        self.assertEqual(by_expression["(d-d)"]["rejected"], "constant_on_training")
        self.assertEqual(by_expression["(t-t)"]["rejected"], "constant_on_training")
        self.assertIn("(d/t)", by_expression)
        self.assertIn("(t/d)", by_expression)
        for record in records:
            if record["decoder_scores"]:
                self.assertEqual(len(record["decoder_scores"]), 274)

    def test_changed_numeric_values_do_not_return_cached_answers(self):
        pairs = tuple((Fraction(d) * Fraction(7, 3), Fraction(t) * Fraction(5, 2)) for d, t in TRAIN_PAIRS)
        rows = observations(pairs)
        self.assert_reconstructs_and_groups(rows, self.run_discovery(rows))

    def test_row_permutation_does_not_choose_a_different_expression(self):
        rows = observations(TRAIN_PAIRS)
        first = self.selected_pair(self.run_discovery(rows))
        shuffled = [rows[i] for i in (5, 2, 0, 4, 1, 3)]
        self.assertEqual(self.selected_pair(self.run_discovery(shuffled)), first)

    def test_all_constant_observations_have_no_admissible_partition(self):
        rows = observations(((2, 1),) * 6)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                discover(rows, Path(directory) / "candidates.jsonl")


if __name__ == "__main__":
    unittest.main()
