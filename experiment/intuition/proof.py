"""Small exact, conservative certificate checker for generated rational programs.

No physical law or target expression occurs in this module.  The invariant and
decoder arrive as ASTs.  Every division must have a nonzero certificate over
the whole declared domain; rational identities are checked by cross products
of exact polynomials.  This is a deliberately small trusted checker, not an
external theorem prover.  An unproved obligation is rejected, even if true.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .expressions import Expr


# Coordinates: observed distance d >= 0, observed duration t > 0,
# and an independently chosen future duration tau > 0.
VARIABLES = ("d", "t", "tau")
Monomial = tuple[int, int, int]
Polynomial = dict[Monomial, Fraction]
ZERO_POWER: Monomial = (0, 0, 0)


def _clean(poly: Polynomial) -> Polynomial:
    return {power: coefficient for power, coefficient in poly.items() if coefficient}


def _add(left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
    result = dict(left)
    for power, coefficient in right.items():
        result[power] = result.get(power, Fraction(0)) + sign * coefficient
    return _clean(result)


def _multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for a, coefficient_a in left.items():
        for b, coefficient_b in right.items():
            power = tuple(x + y for x, y in zip(a, b))
            result[power] = result.get(power, Fraction(0)) + coefficient_a * coefficient_b
    return _clean(result)


def _variable(index: int) -> Polynomial:
    powers = [0, 0, 0]
    powers[index] = 1
    return {tuple(powers): Fraction(1)}


def _strict_sign(poly: Polynomial) -> int:
    """Sufficient sign proof on d>=0,t>0,tau>0; 0 means not proved.

    A monomial is nonnegative throughout the domain.  A monomial with no d
    factor is strictly positive.  Thus coefficients of one strict sign plus
    at least one term without a d factor prove the polynomial's strict sign.
    This is sufficient, not a complete decision procedure for positivity.
    """
    if not poly or not any(power[0] == 0 for power in poly):
        return 0
    coefficients = tuple(poly.values())
    if all(value > 0 for value in coefficients):
        return 1
    if all(value < 0 for value in coefficients):
        return -1
    return 0


def _poly_json(poly: Polynomial) -> dict[str, Any]:
    return {
        "variables": list(VARIABLES),
        "terms": [
            {"powers": list(power), "coefficient": str(coefficient)}
            for power, coefficient in sorted(poly.items())
        ],
    }


def _nonnegative_certificate(value: RationalPolynomial) -> dict[str, Any]:
    """Conservative codomain proof; reconstruction alone cannot provide it."""
    denominator_sign = _strict_sign(value.denominator)
    oriented_numerator = {
        power: coefficient * denominator_sign
        for power, coefficient in value.numerator.items()
    }
    passed = bool(denominator_sign) and all(
        coefficient >= 0 for coefficient in oriented_numerator.values()
    )
    if not passed:
        raise ValueError(
            "Cannot certify future output is nonnegative over d>=0,t>0,tau>0"
        )
    return {
        "claim": "g(phi(d,t),tau) >= 0 over the complete independent-future domain",
        "denominator_strict_sign": denominator_sign,
        "oriented_numerator": _poly_json(oriented_numerator),
        "rule": "strict denominator sign and all oriented numerator coefficients nonnegative",
        "zero_numerator_allowed": True,
        "passed": True,
    }


@dataclass(frozen=True)
class RationalPolynomial:
    numerator: Polynomial
    denominator: Polynomial


def _rf_variable(index: int) -> RationalPolynomial:
    return RationalPolynomial(_variable(index), {ZERO_POWER: Fraction(1)})


def _ast_json(expr: Expr, allowed: set[str]) -> dict[str, Any]:
    kind = getattr(expr, "kind", None)
    args = getattr(expr, "args", None)
    if kind in allowed and args == ():
        return {"var": kind}
    if kind not in {"+", "-", "*", "/"} or not isinstance(args, tuple) or len(args) != 2:
        raise ValueError("Invalid expression or forbidden free variable in proof input")
    return {"op": kind, "args": [_ast_json(arg, allowed) for arg in args]}


def _compile(
    expr: Expr,
    environment: dict[str, RationalPolynomial],
    obligations: list[dict[str, Any]],
    phase: str,
    path: str = "root",
) -> RationalPolynomial:
    if expr.kind in environment:
        return environment[expr.kind]
    left = _compile(expr.args[0], environment, obligations, phase, path + ".0")
    right = _compile(expr.args[1], environment, obligations, phase, path + ".1")
    if expr.kind == "+" or expr.kind == "-":
        numerator = _add(
            _multiply(left.numerator, right.denominator),
            _multiply(right.numerator, left.denominator),
            1 if expr.kind == "+" else -1,
        )
        denominator = _multiply(left.denominator, right.denominator)
    elif expr.kind == "*":
        numerator = _multiply(left.numerator, right.numerator)
        denominator = _multiply(left.denominator, right.denominator)
    elif expr.kind == "/":
        sign = _strict_sign(right.numerator)
        obligation = {
            "phase": phase,
            "ast_path": path,
            "claim": "divisor != 0 on the complete declared domain",
            "divisor_numerator": _poly_json(right.numerator),
            "divisor_denominator": _poly_json(right.denominator),
            "numerator_strict_sign": sign,
            "rule": "same-sign coefficients and a term with d exponent zero",
            "denominator_nonzero": "induction over previously checked AST nodes",
            "passed": sign != 0,
        }
        obligations.append(obligation)
        if not sign:
            raise ValueError(
                f"Cannot certify division totality at {phase}:{path}; "
                "divisor numerator has no strict-sign certificate on d>=0,t>0,tau>0"
            )
        numerator = _multiply(left.numerator, right.denominator)
        denominator = _multiply(left.denominator, right.numerator)
    else:
        raise ValueError(f"Unsupported AST operator: {expr.kind!r}")
    return RationalPolynomial(numerator, denominator)


def _congruence(expr: Expr) -> dict[str, Any]:
    """Proof tree: equal z and equal duration imply equal decoder output."""
    if not expr.args:
        return {"rule": "equal_input", "variable": expr.kind}
    return {
        "rule": "arithmetic_congruence",
        "operator": expr.kind,
        "nonzero_denominators": "certified separately over the complete domain"
        if expr.kind == "/" else "not required",
        "children": [_congruence(arg) for arg in expr.args],
    }


def certify(phi: Expr, decoder: Expr) -> dict[str, Any]:
    """Return a reproducible certificate, or reject an unproved obligation.

    Guarantees, relative to exact rational arithmetic and this small checker:
      * phi is total on rational d>=0,t>0;
      * g(phi(d,t),tau) is total for independently chosen rational tau>0;
      * g(phi(d,t),tau)>=0, so the future output belongs to Distance;
      * g(phi(d,t),t)=d throughout the complete observation domain;
      * kernel(phi) is an equivalence and g descends to its quotient.

    Neither finite samples nor expected answers are inputs to this function.
    """
    phi_ast = _ast_json(phi, {"d", "t"})
    decoder_ast = _ast_json(decoder, {"z", "t"})
    obligations: list[dict[str, Any]] = []
    distance, duration, future_duration = (_rf_variable(i) for i in range(3))
    invariant = _compile(phi, {"d": distance, "t": duration}, obligations, "invariant_totality")
    independent_operation = _compile(
        decoder,
        {"z": invariant, "t": future_duration},
        obligations,
        "future_operation_totality",
    )
    output_nonnegative = _nonnegative_certificate(independent_operation)
    reconstruction = _compile(
        decoder, {"z": invariant, "t": duration}, obligations, "reconstruction_totality"
    )
    difference = _add(
        _multiply(reconstruction.numerator, distance.denominator),
        _multiply(distance.numerator, reconstruction.denominator),
        -1,
    )
    if difference:
        raise ValueError("Reconstruction is not a rational identity: g(phi(d,t),t) != d")
    return {
        "schema_version": 1,
        "status": "proved_by_exact_checker",
        "domain": {
            "scalars": "Q (exact rational numbers)",
            "observation": {"d": ">= 0", "t": "> 0"},
            "independent_future_duration": {"tau": "> 0"},
        },
        "inputs": {"phi": phi_ast, "decoder": decoder_ast},
        "checks": {
            "invariant_total": True,
            "future_operation_total_on_invariant_image": True,
            "output_nonnegative": True,
            "reconstruction_identity": True,
            "equivalence_relation": True,
            "quotient_representation_injective": True,
            "operation_well_defined_on_quotient": True,
        },
        "rational_polynomials": {
            "invariant": {
                "numerator": _poly_json(invariant.numerator),
                "denominator": _poly_json(invariant.denominator),
            },
            "future_operation": {
                "numerator": _poly_json(independent_operation.numerator),
                "denominator": _poly_json(independent_operation.denominator),
            },
            "reconstruction": {
                "numerator": _poly_json(reconstruction.numerator),
                "denominator": _poly_json(reconstruction.denominator),
            },
        },
        "division_obligations": obligations,
        "output_nonnegative_certificate": output_nonnegative,
        "identity_certificate": {
            "claim": "g(phi(d,t),t)=d",
            "rule": "nonzero denominators plus exact cross-multiplication",
            "normalized_cross_difference": _poly_json(difference),
            "is_zero_polynomial": True,
        },
        "equivalence_certificate": {
            "relation": "x ~ y iff phi(x) = phi(y)",
            "template": "kernel of a total function into exact rational equality",
            "reflexivity": "phi(x)=phi(x) by reflexivity of rational equality",
            "symmetry": "phi(x)=phi(y) implies phi(y)=phi(x) by symmetry of equality",
            "transitivity": "phi(x)=phi(y) and phi(y)=phi(w) imply phi(x)=phi(w)",
        },
        "quotient_certificate": {
            "definition": "Type_1 = X / kernel(phi)",
            "representation": "[x] maps to phi(x), with codomain Im(phi)",
            "well_defined": "x ~ y implies phi(x)=phi(y) by definition",
            "injective": "phi(x)=phi(y) implies [x]=[y] by definition of kernel(phi)",
            "surjective_onto_image": "each element of Im(phi) has a preimage in X",
        },
        "operation_certificate": {
            "definition": "Travel([x],tau)=g(phi(x),tau)",
            "free_variables": ["z", "t"],
            "reason": "equal invariant representations and equal duration give equal output",
            "congruence_tree": _congruence(decoder),
            "totality": "independent tau checked above, rather than only observed t",
        },
        "trusted_basis": [
            "Python integer and Fraction arithmetic",
            "this polynomial implementation and AST interpreter agreement",
            "ordered-field sign rules on the declared rational domain",
            "equality axioms and the generic kernel/quotient construction",
        ],
        "limitations": [
            "strict-sign denominator reasoning is sound but incomplete; unproved means rejected",
            "this is not an independently verified theorem prover",
            "no conclusion about physical generalization follows from rational identities",
        ],
    }
