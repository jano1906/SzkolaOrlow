import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
from typing import List, Callable, Any, Tuple


Point: type = Any
Label = int
Dpoint = Tuple[Point, Label]


def plot2D_xc(X: List[Point], c: List[Label] = None):
    xs = [p[0] for p in X]
    ys = [p[1] for p in X]
    return plt.scatter(xs, ys, c=c)


def plot2D_data(data: List[Dpoint]):
    xs = [p[0][0] for p in data]
    ys = [p[0][1] for p in data]
    cs = [p[1] for p in data]
    return plt.scatter(xs, ys, c=cs)


def plot3D_xc(X: List[Point], c: List[Label] = None):
    xs = [x[0] for x in X]
    ys = [x[1] for x in X]
    zs = [x[2] for x in X]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax.scatter(xs, ys, zs, c=c)


def plot3D_data(data: List[Dpoint]):
    xs = [x[0][0] for x in data]
    ys = [x[0][1] for x in data]
    zs = [x[0][2] for x in data]
    cs = [x[1] for x in data]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax.scatter(xs, ys, zs, c=cs)


def _printFigures(Xs: List[Point], centers: List[Point], labels: List[Label], rs: List[float],
                  norm: Callable, default_label: Label) -> List[Dpoint]:
    rng = default_rng()
    res: List[Dpoint] = list()
    for p in Xs:
        p = np.array(p)
        in_regions: List[Label] = list()
        for (c, l, r) in zip(centers, labels, rs):
            c = np.array(c)
            if norm(p - c) <= r:
                in_regions.append(l)
        dpoint: Dpoint
        if len(in_regions) > 0:
            l = rng.integers(0, len(in_regions))
            dpoint = (p, in_regions[l])
        else:
            dpoint = (p, default_label)
        res.append(dpoint)
    return res


def uniform_random_figures(NO_CENTERS=3, RADIUS_SPREAD=0.3, D=2):
    rng = default_rng()
    centers = rng.uniform(0, 1, size=(NO_CENTERS, D))
    labels = [1] * NO_CENTERS
    rs = rng.uniform(0, RADIUS_SPREAD, NO_CENTERS)
    return centers, labels, rs


def normal_random_figures(NO_CENTERS=3, RADIUS_SPREAD=0.05, MU=0.5, SIG=1., D=2):
    rng = default_rng()
    centers = rng.normal(MU, SIG, size=(NO_CENTERS, D))
    labels = [1] * NO_CENTERS
    rs = rng.uniform(0, RADIUS_SPREAD, NO_CENTERS)
    return centers, labels, rs


def uniform(centers: List[Point], labels: List[Label], rs: List[float],
            norm: Callable = np.linalg.norm, default_label: Label = -1, n: int = 1000, D=2) -> List[Dpoint]:
    assert len(centers) == len(labels) == len(rs), f"centers: {len(centers)}, labels: {len(labels)}, rs: {len(rs)}"
    rng = default_rng()
    Xs = rng.uniform(0, 1, size=(n, D))
    return _printFigures(Xs, centers, labels, rs, norm, default_label)


def normal(centers: List[Point], labels: List[Label], rs: List[float],
           norm: Callable = np.linalg.norm, default_label: Label = -1, n: int = 1000,
           MU=0., SIG=1., D=2) -> List[Dpoint]:
    rng = default_rng()
    Xs = rng.normal(MU, SIG, size=(n, D))
    return _printFigures(Xs, centers, labels, rs, norm, default_label)


def noise(data: List[Dpoint], NOISE=0.1):
    rng = default_rng()
    ps = rng.uniform(0, 1, len(data))
    return [x if p >= NOISE else (x[0], -x[1]) for (x, p) in zip(data, ps)]
