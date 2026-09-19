"""Generates the tightly-converged reference optimum in logistic-reference.json.

Why this exists
---------------
`LbfgsMinimizerTest` needs a logistic problem whose optimum is known to far better
precision than the optimiser under test will reach, so the test pins correctness rather
than pinning whatever L-BFGS currently produces.

The optimum here is found by Newton-Raphson on the exact Hessian. That is a genuinely
different algorithm from L-BFGS -- second order with an explicitly formed and inverted
Hessian, against a quasi-Newton method that never forms one -- so agreement between them
is evidence, not a tautology. Newton converges quadratically, and the run below drives the
gradient to ~1e-16, which is the same guarantee scipy's `gtol=1e-12` would give.

Do NOT regenerate this against `sklearn.LogisticRegression`'s returned coefficients. At its
default `tol = 1e-4` it stops roughly 4e-4 short of the optimum, and a correct L-BFGS then
"fails" the test by being more accurate than the reference.

Run:  python3 logistic_reference.py > logistic-reference.json
Needs only the standard library.
"""

import json
from math import exp, log1p

FEATURES = 3
SAMPLES = 24
INVERSE_REGULARIZATION = 2.0


def lcg(seed):
    """A plain LCG, so the dataset is reproducible without numpy."""
    state = seed
    while True:
        state = (1103515245 * state + 12345) % (1 << 31)
        yield state / (1 << 31)


def dataset():
    """A deliberately non-separable problem, so the optimum is interior and unique."""
    rand = lcg(seed=20240919)
    rows, labels = [], []
    for i in range(SAMPLES):
        # Multiples of 2^-10: exactly representable in float32 as well as float64, so the
        # matrix the Kotlin test builds holds bit-identical values and the two solvers are
        # comparing answers to the same problem rather than to two roundings of it.
        x = [round((2.0 * next(rand) - 1.0) * 1024.0) / 1024.0 for _ in range(FEATURES)]
        # A noisy linear rule: the sign of a fixed combination, flipped on every 7th row so
        # no weight vector separates the data and the L2 term has a finite minimiser.
        score = 1.5 * x[0] - 0.75 * x[1] + 0.5 * x[2]
        positive = score > 0.0
        if i % 7 == 3:
            positive = not positive
        rows.append(x)
        labels.append(1 if positive else -1)
    return rows, labels


def logistic(z):
    if z >= 0.0:
        return 1.0 / (1.0 + exp(-z))
    t = exp(z)
    return t / (1.0 + t)


def softplus(z):
    if z > 0.0:
        return z + log1p(exp(-z))
    return log1p(exp(z))


def value_gradient_hessian(params, rows, labels):
    size = FEATURES + 1
    value = 0.5 * sum(w * w for w in params[:FEATURES])
    gradient = [params[j] if j < FEATURES else 0.0 for j in range(size)]
    hessian = [[1.0 if (r == c and r < FEATURES) else 0.0 for c in range(size)] for r in range(size)]
    for x, y in zip(rows, labels):
        augmented = x + [1.0]
        z = sum(params[j] * augmented[j] for j in range(size))
        value += INVERSE_REGULARIZATION * softplus(-y * z)
        slope = -y * logistic(-y * z)
        p = logistic(z)
        curvature = p * (1.0 - p)
        for r in range(size):
            gradient[r] += INVERSE_REGULARIZATION * slope * augmented[r]
            for c in range(size):
                hessian[r][c] += INVERSE_REGULARIZATION * curvature * augmented[r] * augmented[c]
    return value, gradient, hessian


def solve(matrix, rhs):
    """Gaussian elimination with partial pivoting on a 4x4 system."""
    size = len(rhs)
    a = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(a[r][col]))
        a[col], a[pivot] = a[pivot], a[col]
        for r in range(size):
            if r == col:
                continue
            factor = a[r][col] / a[col][col]
            for c in range(col, size + 1):
                a[r][c] -= factor * a[col][c]
    return [a[r][size] / a[r][r] for r in range(size)]


def main():
    rows, labels = dataset()
    params = [0.0] * (FEATURES + 1)
    for _ in range(100):
        value, gradient, hessian = value_gradient_hessian(params, rows, labels)
        step = solve(hessian, gradient)
        params = [params[j] - step[j] for j in range(len(params))]
        if max(abs(g) for g in gradient) < 1e-15:
            break
    value, gradient, _ = value_gradient_hessian(params, rows, labels)
    print(json.dumps({
        "inverseRegularization": INVERSE_REGULARIZATION,
        "featureCount": FEATURES,
        "rows": rows,
        "labels": labels,
        "optimum": params,
        "optimalValue": value,
        "gradientNormAtOptimum": max(abs(g) for g in gradient),
    }, indent=2))


if __name__ == "__main__":
    main()
