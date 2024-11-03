"""
Collection of the core mathematical operators used throughout the code base.

P.S. ChatGPT was used to generate docstring for functions
"""

import math
from typing import Callable, Iterable

def mul(x: float, y: float) -> float:
    """
    Returns the product of two numbers x and y.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Product of x and y.
    """
    return x * y

def id(x: float) -> float:
    """
    Returns the identity of x (i.e., x itself).

    Parameters:
    - x: Input value.

    Returns:
    - x itself.
    """
    return x

def add(x: float, y: float) -> float:
    """
    Returns the sum of two numbers x and y.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Sum of x and y.
    """
    return x + y

def neg(x: float) -> float:
    """
    Returns the negation of x.

    Parameters:
    - x: Input value.

    Returns:
    - Negated value of x.
    """
    return -x

def lt(x: float, y: float) -> bool:
    """
    Returns True if x is less than y, otherwise False.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Boolean indicating whether x < y.
    """
    return x < y

def eq(x: float, y: float) -> bool:
    """
    Returns True if x is equal to y, otherwise False.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Boolean indicating whether x == y.
    """
    return x == y

def max(x: float, y: float) -> float:
    """
    Returns the maximum of two numbers x and y.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Maximum of x and y.
    """
    return x if x > y else y

def min(x: float, y: float) -> float:
    """
    Returns the minimum of two numbers x and y.

    Parameters:
    - x: First number.
    - y: Second number.

    Returns:
    - Minimum of x and y.
    """
    return x if x < y else y

def is_close(input: float, other: float, tol: float = 1e-2) -> bool:
    """
    Checks if two numbers are within a tolerance 'tol' of each other.

    This function implements:
    f(x, y) = |x - y| < tol

    Parameters:
    - input: First number for comparison.
    - other: Second number for comparison.
    - tol: Tolerance value. Default is 1e-2.

    Returns:
    - True if |input - other| < tol, otherwise False.
    """
    return abs(input - other) < tol

def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid of x with an optimized formula to avoid overflow.

    This function implements:
    f(x) = 1 / (1 + e^(-x)) if x >= 0, else e^x / (1 + e^x)

    Parameters:
    - x: The input value.

    Returns:
    - Sigmoid value of x.
    """
    return 1.0 / (1.0 + math.exp(-x)) if x > 0 else math.exp(x) / (1.0 + math.exp(x))

def relu(x: float) -> float:
    """
    Applies the ReLU (Rectified Linear Unit) function on x.

    Parameters:
    - x: The input value.

    Returns:
    - x if x > 0, otherwise 0.
    """
    return x if x > 0 else 0.0

def log(x: float) -> float:
    """
    Returns the natural logarithm of x.

    Parameters:
    - x: The input value.

    Returns:
    - Natural logarithm of x.
    """
    return math.log(x)

def exp(x: float) -> float:
    """
    Returns e raised to the power of x.

    Parameters:
    - x: The input value.

    Returns:
    - Exponential of x.
    """
    return math.exp(x)

def log_back(x: float, d: float) -> float:
    """
    Calculates the derivative of log(x) with respect to x, given an initial derivative d.

    Parameters:
    - x: Input value.
    - d: Initial derivative with respect to the output of log(x).

    Returns:
    - The backpropagated derivative of log with respect to x.
    """
    return d / x

def inv(x: float) -> float:
    """
    Returns the multiplicative inverse of x.

    Parameters:
    - x: Input value.

    Returns:
    - 1 / x.
    """
    return 1.0 / x

def inv_back(x: float, d: float) -> float:
    """
    Calculates the derivative of 1/x with respect to x, given an initial derivative d.

    Parameters:
    - x: Input value.
    - d: Initial derivative with respect to the output of 1/x.

    Returns:
    - The backpropagated derivative of the inverse with respect to x.
    """
    return -d / (x**2)

def relu_back(x: float, d: float) -> float:
    """
    Returns the backpropagated derivative for ReLU given x and an initial derivative d.

    Parameters:
    - x: Input value.
    - d: Initial derivative with respect to the output of ReLU.

    Returns:
    - d if x > 0, otherwise 0.
    """
    return d if x > 0 else 0.0

def map(fn: Callable[[float], float], iter: Iterable[float]) -> Iterable[float]:
    """
    Applies function fn to each element in iter and returns a list of results.

    Parameters:
    - fn: Function to apply to each element.
    - iter: Iterable of values to apply fn on.

    Returns:
    - List of results after applying fn to each element in iter.
    """
    return [fn(x) for x in iter]

def zipWith(fn: Callable[[float, float], float], iter1: Iterable[float], iter2: Iterable[float]) -> Iterable[float]:
    """
    Applies a binary function fn to pairs from iter1 and iter2, returning a list of results.

    Parameters:
    - fn: Function to apply to pairs of elements from iter1 and iter2.
    - iter1: First iterable of values.
    - iter2: Second iterable of values.

    Returns:
    - List of results after applying fn to each pair from iter1 and iter2.
    """
    return [fn(x, y) for x, y in zip(iter1, iter2)]

def reduce(fn: Callable[[float, float], float], iter: Iterable[float], init: float) -> float:
    """
    Reduces an iterable to a single value by iteratively applying a binary function fn.

    Parameters:
    - fn: Function to apply.
    - iter: Iterable of values to reduce.
    - init: Initial value for the reduction.

    Returns:
    - Reduced single value.
    """
    result = init
    for x in iter:
        result = fn(result, x)
    return result

def negList(lst: Iterable[float]) -> Iterable[float]:
    """
    Returns a list where each element of lst is negated.

    Parameters:
    - lst: Iterable of values to negate.

    Returns:
    - List of negated values.
    """
    return map(neg, lst)

def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """
    Returns a list where each element is the sum of elements from lst1 and lst2 at corresponding positions.

    Parameters:
    - lst1: First iterable of values.
    - lst2: Second iterable of values.

    Returns:
    - List of element-wise sums of lst1 and lst2.
    """
    return zipWith(add, lst1, lst2)

def sum(lst: Iterable[float]) -> float:
    """
    Returns the sum of all elements in lst.

    Parameters:
    - lst: Iterable of values to sum.

    Returns:
    - Sum of all values in lst.
    """
    return reduce(add, lst, 0)

def prod(lst: Iterable[float]) -> float:
    """
    Returns the product of all elements in lst.

    Parameters:
    - lst: Iterable of values to multiply.

    Returns:
    - Product of all values in lst.
    """
    return reduce(mul, lst, 1)
