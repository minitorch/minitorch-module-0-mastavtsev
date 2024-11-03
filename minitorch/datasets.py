import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N):
    """
    Generates N random points in a 2D space with coordinates between 0 and 1.

    Args:
        N (int): Number of points to generate.

    Returns:
        List[Tuple[float, float]]: A list of tuples, each representing the (x_1, x_2) coordinates of a point.
    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    """
    A class representing a 2D dataset with points and labels.

    Attributes:
        N (int): Number of points in the dataset.
        X (List[Tuple[float, float]]): List of points represented as (x_1, x_2) coordinates.
        y (List[int]): List of labels (0 or 1) corresponding to each point.
    """
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N):
    """
    Generates a simple linear classification dataset.

    Points are labeled as 1 if x_1 < 0.5, and 0 otherwise.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on a simple linear separation.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N):
    """
    Generates a diagonal classification dataset.

    Points are labeled as 1 if x_1 + x_2 < 0.5, and 0 otherwise.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on a diagonal boundary.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N):
    """
    Generates a split classification dataset.

    Points are labeled as 1 if x_1 < 0.2 or x_1 > 0.8, and 0 otherwise.
    This creates two separated regions for class 1 along the x-axis.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on two split regions.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N):
    """
    Generates an XOR classification dataset.

    Points are labeled as 1 if they lie in opposing quadrants (x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5),
    and 0 otherwise. This creates a non-linear boundary that resembles an XOR pattern.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on an XOR pattern.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 and x_2 > 0.5 or x_1 > 0.5 and x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N):
    """
    Generates a circular classification dataset.

    Points are labeled as 1 if they lie outside a circle of radius sqrt(0.1) centered at (0.5, 0.5),
    and 0 otherwise. This creates a circular boundary separating the two classes.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on a circular boundary.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = x_1 - 0.5, x_2 - 0.5
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N):
    """
    Generates a spiral classification dataset.

    Points are arranged in two intertwined spirals, with one labeled as 0 and the other as 1.
    This dataset is challenging to classify with a linear model due to the spiral shape of the boundary.

    Args:
        N (int): Number of points to generate.

    Returns:
        Graph: A dataset with points and binary labels based on a spiral pattern.
    """

    def x(t):
        return t * math.cos(t) / 20.0

    def y(t):
        return t * math.sin(t) / 20.0
    X = [(x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N //
        2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    X = X + [(y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) /
        (N // 2))) + 0.5) for i in range(5 + 0, 5 + N // 2)]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {'Simple': simple, 'Diag': diag, 'Split': split, 'Xor': xor,
    'Circle': circle, 'Spiral': spiral}
