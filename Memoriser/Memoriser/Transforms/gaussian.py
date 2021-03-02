import numpy as np
from numpy.linalg import norm
from typing import List, Callable, Any, Tuple
import scipy.stats as ss

Point: type = Any
Label = int
Dpoint = Tuple[Point, Label]


D = 2
MU = 0.
SIG = 1.
MEAN = np.full(D, MU)
COV = np.diag(np.full(D, SIG))


def cdf(x):
    return abs(ss.norm.cdf(x)-0.5)


def ppf(x):
    return abs(ss.norm.ppf(x+0.5))


def update():
    global MEAN, COV
    MEAN = np.full(D, MU)
    COV = np.diag(np.full(D, SIG))


def T(p: Point) -> Point:
    v = p-MEAN
    if norm(v) != 0:
        p = MEAN + v * ppf(norm(v)) / norm(v)
    return p


def Tinv(p: Point) -> Point:
    v = p - MEAN
    if norm(v) != 0:
        p = MEAN + v * cdf(norm(v)) / norm(v)
    return p
