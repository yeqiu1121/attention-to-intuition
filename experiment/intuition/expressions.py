"""Domain-neutral arithmetic ASTs. No constants, target formula, or target type.

The full ordered grammar is enumerated, without algebraic pruning. Fractions
are the only numeric representation. This module has no observation fixtures.
"""
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

OPS = ("+", "-", "*", "/")


@dataclass(frozen=True)
class Expr:
    kind: str
    args: tuple = ()


def size(expr):
    return 1 + sum(size(arg) for arg in expr.args)


def render(expr):
    if not expr.args:
        return expr.kind
    return f"({render(expr.args[0])}{expr.kind}{render(expr.args[1])})"


@lru_cache(maxsize=None)
def enumerate_exprs(variables, max_nodes=5):
    if not variables or len(set(variables)) != len(variables):
        raise ValueError("Distinct variable names required")
    if max_nodes < 1 or max_nodes > 5:
        raise ValueError("V0.1 search bound is 1..5 nodes")
    by_size = {1: tuple(Expr(v) for v in variables)}
    for count in range(2, max_nodes + 1):
        expressions = []
        for left_count in range(1, count - 1):
            right_count = count - 1 - left_count
            for op in OPS:
                for left in by_size.get(left_count, ()):
                    for right in by_size.get(right_count, ()):
                        expressions.append(Expr(op, (left, right)))
        by_size[count] = tuple(expressions)
    return tuple(expr for count in range(1, max_nodes + 1) for expr in by_size[count])


def evaluate(expr, env):
    if not expr.args:
        return Fraction(env[expr.kind])
    left, right = (evaluate(arg, env) for arg in expr.args)
    if expr.kind == "+":
        return left + right
    if expr.kind == "-":
        return left - right
    if expr.kind == "*":
        return left * right
    if expr.kind == "/":
        return left / right
    raise ValueError("Unknown operator")


def to_json(expr):
    if not expr.args:
        return {"var": expr.kind}
    return {"op": expr.kind, "args": [to_json(arg) for arg in expr.args]}


def from_json(data):
    if not isinstance(data, dict):
        raise ValueError("AST node must be an object")
    if set(data) == {"var"}:
        if data["var"] not in ("d", "t", "z"):
            raise ValueError("Unknown variable")
        return Expr(data["var"])
    if set(data) != {"op", "args"} or data["op"] not in OPS:
        raise ValueError("Invalid AST operator or fields")
    if not isinstance(data["args"], list) or len(data["args"]) != 2:
        raise ValueError("Binary AST requires two children")
    expr = Expr(data["op"], tuple(from_json(arg) for arg in data["args"]))
    if size(expr) > 5:
        raise ValueError("AST exceeds five-node V0.1 bound")
    return expr
