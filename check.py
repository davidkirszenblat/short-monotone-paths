"""Numerical verification of the auxiliary construction.

For random linear programs min c.x s.t. A0 x <= b, builds the monotone path
from the apex to the optimum and checks: the path exists, has at most
(n-d)(n-d+1)/2 edges, descends the objective (|y_{n-d}|,...,|y_1|,c.x)
lexicographically, and visits only vertices of the split polyhedron.
Unbounded or degenerate instances are skipped.
"""

import numpy as np
from scipy.optimize import linprog

TOL = 1e-8


def verify(seed, d, n):
    rng = np.random.default_rng(seed)
    A0 = rng.standard_normal((n, d))
    b = 1 + A0 @ rng.standard_normal(d)  # feasible: RHS built around an interior point
    c = rng.standard_normal(d)
    A = np.hstack([A0, np.eye(n)[:, : n - d]])  # tilt the first n-d constraints

    sol = linprog(c, A_ub=A0, b_ub=b, bounds=(None, None))
    if not sol.success:
        return "skip"
    x = {d: np.concatenate([sol.x, np.zeros(n - d)])}
    B = {d: list(np.flatnonzero(np.abs(A @ x[d] - b) < TOL))}
    if len(B[d]) != d:
        return "skip"

    # walk from the optimum to the apex, adding one tight constraint per step
    for i in range(d + 1, n + 1):
        for j in set(range(n)) - set(B[i - 1]):
            basis = B[i - 1] + [j]
            try:
                v = np.linalg.solve(A[np.ix_(basis, range(i))], b[basis])
            except np.linalg.LinAlgError:
                continue
            v = np.concatenate([v, np.zeros(n - i)])
            if np.all(A @ v - b <= TOL):
                B[i], x[i] = basis, v
                break
        else:
            return "no lift at step %d" % i

    # subdivide each segment where a slack changes sign
    path = []
    for i in range(d, n):
        y0, y1 = x[i][d:], x[i + 1][d:]
        k = np.flatnonzero(y0 * y1 < -TOL)
        s = np.sort(-y0[k] / (y1[k] - y0[k]))
        path += [x[i]] + [(1 - t) * x[i] + t * x[i + 1] for t in s]
    path.append(x[n])

    if len(path) - 1 > (n - d) * (n - d + 1) // 2:
        return "bound exceeded"

    key = lambda p: np.concatenate([np.abs(p[d:])[::-1], [c @ p[:d]]])
    for p, q in zip(path, path[1:]):
        delta = key(q) - key(p)
        nz = np.flatnonzero(np.abs(delta) > TOL)
        if len(nz) == 0 or delta[nz[0]] < 0:
            return "not monotone"

    # tight constraints of the split polyhedron: rows of A, plus two per zero
    # slack (y+ = y- = 0) and one per nonzero slack
    for p in path:
        tight = (np.sum(np.abs(A @ p - b) < TOL)
                 + np.sum(np.abs(p[d:]) < TOL) * 2
                 + np.sum(np.abs(p[d:]) >= TOL))
        if tight != 2 * n - d:
            return "non-vertex on path"

    return len(path) - 1


if __name__ == "__main__":
    for d, n in [(3, 10), (5, 15), (5, 20), (8, 20), (10, 25)]:
        results = [verify(seed, d, n) for seed in range(200)]
        edges = [r for r in results if isinstance(r, int)]
        errors = [r for r in results if r not in edges and r != "skip"]
        print("(%d,%d): %d verified, %d skipped, max edges %d, bound %d%s"
              % (d, n, len(edges), results.count("skip"), max(edges),
                 (n - d) * (n - d + 1) // 2,
                 "; FAILURES: " + ", ".join(set(errors)) if errors else ""))
