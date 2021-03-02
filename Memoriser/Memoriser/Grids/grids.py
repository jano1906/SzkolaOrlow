import numpy as np
from typing import List, Any
from ..DataGenerator import figures as dgf
from math import floor


def _cube(grid: List[dgf.Point], D: int, d: int, step, pts_on_side: int):
    if d == D:
        return
    if d == 0:
        start = np.zeros(D)
        for i in range(pts_on_side):
            grid.append(start.copy())
            start[d] += step
        _cube(grid, D, d + 1, step, pts_on_side)
    else:
        grid2 = list()
        for g in grid:
            start = np.zeros(D)
            for i in range(pts_on_side-1):
                start[d] += step
                grid2.append(start + g)
        grid += grid2
        _cube(grid, D, d + 1, step, pts_on_side)


def cube(grid: List[dgf.Point], n: int, D: int = 2, side_len: float = 1.):
    pts_on_side = n**(1./D)
    step = side_len/(pts_on_side-1)
    _cube(grid, D, 0, step, floor(pts_on_side))
    return 1./step


def ballV(D: int = 2, r: float = 1.):
    vols = np.empty(D+1)
    vols[0] = 1
    vols[1] = 2*r
    r2 = r**2
    for i in range(2, D+1):
        vols[i] = (vols[i-2]*2*np.pi*r2)/i
    return vols[D]


def _ball(grid: List[dgf.Point], D: int, d: int, step, r2: float):
    if d == D:
        return
    if d == 0:
        start = np.zeros(D)
        norm2 = 0.
        while norm2 <= r2:
            grid.append(start.copy())
            norm2 += step*(2*start[d] + step)
            start[d] += step
        _ball(grid, D, d + 1, step, r2)
    else:
        grid2 = list()
        for g in grid:
            norm2 = np.dot(g, g)
            norm2 += step*(2*g[d] + step)
            start = np.zeros(D)
            while norm2 <= r2:
                start[d] += step
                grid2.append(start + g)
                norm2 += step * (2 * start[d] + step)
        grid += grid2
        _ball(grid, D, d + 1, step, r2)


def ball(grid: List[dgf.Point], n: int, center: Any = None, D: int = 2, r: float = 1.):
    if center is None:
        center = np.full(D, 0)
    r2 = r**2
    step = (ballV(D, r) ** (1./D))/(n**(1./D) - 1)
    _ball(grid, D, 0, step, r2)
    for d in range(D):
        for g in grid.copy():
            if g[d] == 0:
                continue
            g = g.copy()
            g[d] *= -1
            grid.append(g)
    for g in grid:
        g += center
    return 1./step


def scale(s: float, grid: List[dgf.Point]):
    return np.round(np.array(grid)*s)
