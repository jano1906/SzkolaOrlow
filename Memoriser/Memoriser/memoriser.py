import numpy as np
from typing import List, Callable, Tuple, Any, Dict, ByteString
from .Grids import grids as gs

Point: type = Any
Label = int
Hash = str
Table = Dict[ByteString, Label]


def _id(x):
    return x


def build_simplex(p: Point, D: int = 2):
    vert0 = np.round(p)
    projs = list()
    for i in range(D):
        if vert0[i] < p[i]:
            projs.append(-1)
        else:
            projs.append(1)
    simplex = [vert0]
    for i in range(D):
        verti = vert0.copy()
        verti[i] -= projs[i]
        simplex.append(verti)
    return simplex


def AInterpolation(point: Point, LU_table: Table, scale: float, D: int = 2) -> float:
    p = point.copy()
    p *= scale
    simplex = build_simplex(p, D)
    vert0 = simplex[0]

    vert_in: Point = vert0
    for vert in simplex:
        if vert.tostring() in LU_table:
            vert_in = vert
            break
    if vert_in.tostring() not in LU_table:
        return -1
    #print(simplex, vert_in)
    res = 0.
    weight = 0.
    p -= vert0
    for i in range(1, D):
        w = np.abs(p[i])
        weight += w
        if simplex[i].tostring() not in LU_table:
            simplex[i] = vert_in
        res += w * LU_table[simplex[i].tostring()]
    if vert0.tostring() not in LU_table:
        vert0 = vert_in
    res += np.abs(1 - weight) * LU_table[vert0.tostring()]
    return res


class Memo:
    def __init__(self, T: Callable = _id, Tinv: Callable = _id):
        self.LU_table: Table = dict()
        self.scale: float = -1.
        self.T: Callable = T
        self.Tinv: Callable = Tinv

    def memo(self, scale, grid, predictor):
        self.scale = scale
        X_table: List[Point] = list(map(self.T, grid))
        Y_table: List[Label] = predictor.predict(X_table)
        X_table = list(map(self.Tinv, X_table))
        X_table = gs.scale(scale, X_table)
        hashes = [x.tostring() for x in X_table]
        self.LU_table = dict(zip(hashes, Y_table))
        return X_table, Y_table

    def pred(self, Xs: List[Point]) -> List[Label]:
        return [int(np.sign(AInterpolation(self.Tinv(x), self.LU_table, self.scale))) for x in Xs]

    def score(self, X_test, Y_test):
        succ = 0
        Y_pred = self.pred(X_test)
        for t in zip(Y_pred, Y_test):
            if t[0] == t[1]:
                succ += 1
        return succ / len(X_test)
