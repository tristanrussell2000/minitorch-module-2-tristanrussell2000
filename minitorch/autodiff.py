from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    plus_epsilon = vals[:arg] + (vals[arg] + (epsilon / 2),) + vals[arg + 1 :]
    minus_epsilon = vals[:arg] + (vals[arg] - (epsilon / 2),) + vals[arg + 1 :]
    return (f(*plus_epsilon) - f(*minus_epsilon)) / epsilon


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    marked: set[int] = set()
    ordered: list[Variable] = []
    def visit(s: Variable) -> None:
        if s.unique_id in marked or s.is_constant():
            return
        for v in s.parents:
            visit(v)
        marked.add(s.unique_id)
        ordered.insert(0, s)
    visit(variable)
    return ordered


def backpropagate(variable: Variable, deriv: Any) -> None:
    visit_list = topological_sort(variable)
    int_derivs: dict[int, float] = {variable.unique_id: deriv}
    def add_deriv(id: int, d: float) -> None:
        int_derivs[id] = int_derivs.get(id, 0) + d
    for v in visit_list:
        if v.is_leaf():
            v.accumulate_derivative(int_derivs[v.unique_id])
        else:
            vars_and_derivs = v.chain_rule(int_derivs[v.unique_id])
            for v2, d2 in vars_and_derivs:
                add_deriv(v2.unique_id, d2)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
