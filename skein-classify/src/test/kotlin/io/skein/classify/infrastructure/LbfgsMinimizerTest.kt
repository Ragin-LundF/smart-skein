package io.skein.classify.infrastructure

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class LbfgsMinimizerTest {

    /** `sum (x_i - centre_i)^2`, whose minimum is exactly `centre` and whose curvature is constant. */
    private class Quadratic(private val centre: DoubleArray) : DifferentiableFunction {

        override val dimension: Int = centre.size

        override fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double {
            var total = 0.0
            point.indices.forEach { i ->
                val offset = point[i] - centre[i]
                total += offset * offset
                gradient[i] = 2.0 * offset
            }
            return total
        }
    }

    /**
     * Rosenbrock's function: `(1 - x)^2 + 100 (y - x^2)^2`, minimum 0 at `(1, 1)`.
     *
     * The standard quasi-Newton stress test. Its minimum sits at the bottom of a long curved
     * valley, so steepest descent zig-zags across it almost indefinitely; only a method that
     * accumulates curvature information gets down it in a sane number of iterations.
     */
    private class Rosenbrock : DifferentiableFunction {

        override val dimension: Int = 2

        override fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double {
            val x = point[0]
            val y = point[1]
            val residual = y - x * x
            gradient[0] = -2.0 * (1.0 - x) - 400.0 * x * residual
            gradient[1] = 200.0 * residual
            return (1.0 - x) * (1.0 - x) + 100.0 * residual * residual
        }
    }

    /** A function with no minimum at all: the line search can never find a decrease that stops. */
    private class Linear : DifferentiableFunction {

        override val dimension: Int = 1

        override fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double {
            gradient[0] = 1.0
            return point[0]
        }
    }

    @Test
    internal fun `finds the minimum of a quadratic with a known solution`() {
        val centre = doubleArrayOf(3.0, -2.0, 0.5)

        val result = LbfgsMinimizer().minimize(function = Quadratic(centre = centre))

        assertTrue(actual = result.converged)
        centre.indices.forEach { i ->
            assertEquals(expected = centre[i], actual = result.point[i], absoluteTolerance = 1e-8)
        }
        assertEquals(expected = 0.0, actual = result.value, absoluteTolerance = 1e-14)
    }

    @Test
    internal fun `descends the Rosenbrock valley to its minimum`() {
        val result = LbfgsMinimizer(gradientTolerance = 1e-10, maxIterations = 2000)
            .minimize(function = Rosenbrock(), start = doubleArrayOf(-1.2, 1.0))

        assertTrue(actual = result.converged, message = "did not converge in ${result.iterations} iterations")
        assertEquals(expected = 1.0, actual = result.point[0], absoluteTolerance = 1e-6)
        assertEquals(expected = 1.0, actual = result.point[1], absoluteTolerance = 1e-6)
    }

    /**
     * The test the plan calls the one that matters: agreement with an independently computed
     * optimum, not with a previous run of this optimiser.
     *
     * The reference comes from a Newton solve that drives the gradient to ~1e-16 (see
     * `logistic_reference.py`), so this asserts optimality rather than reproducing a stopping
     * point. Deliberately *not* pinned to `sklearn.LogisticRegression`'s coefficients: at its
     * default `tol = 1e-4` it stops around 4e-4 short, and a correct L-BFGS would fail such a test
     * by being more accurate.
     */
    @Test
    internal fun `reaches the independently computed optimum of the reference logistic problem`() {
        val problem = LogisticReferenceProblem.load()

        val result = LbfgsMinimizer(gradientTolerance = 1e-10, maxIterations = 500)
            .minimize(function = problem.objective())

        assertTrue(actual = result.converged, message = "stopped after ${result.iterations} iterations")
        problem.optimum.indices.forEach { j ->
            assertEquals(
                expected = problem.optimum[j],
                actual = result.point[j],
                absoluteTolerance = 1e-7,
                message = "coefficient $j",
            )
        }
        assertEquals(expected = problem.optimalValue, actual = result.value, absoluteTolerance = 1e-10)
    }

    @Test
    internal fun `converges tighter than scikit-learn's default stopping point`() {
        val problem = LogisticReferenceProblem.load()

        val result = LbfgsMinimizer(gradientTolerance = 1e-10).minimize(function = problem.objective())

        // The claim in the plan, made checkable: the fit lands far inside the 1e-4 tolerance that
        // a default scikit-learn run stops at.
        assertTrue(
            actual = result.gradientNorm < 1e-10,
            message = "gradient norm was ${result.gradientNorm}",
        )
    }

    @Test
    internal fun `returns the starting point when it is already optimal`() {
        val centre = doubleArrayOf(1.0, 2.0)

        val result = LbfgsMinimizer().minimize(function = Quadratic(centre = centre), start = centre)

        assertTrue(actual = result.converged)
        assertEquals(expected = 0, actual = result.iterations)
    }

    @Test
    internal fun `never mutates the caller's starting vector`() {
        val start = doubleArrayOf(9.0, 9.0, 9.0)

        LbfgsMinimizer().minimize(function = Quadratic(centre = doubleArrayOf(0.0, 0.0, 0.0)), start = start)

        assertTrue(actual = start.all { entry -> entry == 9.0 })
    }

    @Test
    internal fun `reports non-convergence when the iteration budget runs out`() {
        val result = LbfgsMinimizer(maxIterations = 2, gradientTolerance = 1e-14)
            .minimize(function = Rosenbrock(), start = doubleArrayOf(-1.2, 1.0))

        assertFalse(actual = result.converged)
        assertEquals(expected = 2, actual = result.iterations)
    }

    @Test
    internal fun `stops and reports the best point when no step can decrease the objective`() {
        // An unbounded objective: every backtrack still decreases it, so the search walks until the
        // step underflows, and the minimiser must report rather than loop.
        val result = LbfgsMinimizer(maxIterations = 50).minimize(function = Linear())

        assertFalse(actual = result.converged)
        assertTrue(actual = result.value.isFinite())
    }

    @Test
    internal fun `converges from a distant start on an ill-scaled quadratic`() {
        // Curvature spanning six orders of magnitude: this is what the initial Hessian scaling in
        // the two-loop recursion exists for.
        val illScaled = object : DifferentiableFunction {
            override val dimension: Int = 3
            private val scales = doubleArrayOf(1e-3, 1.0, 1e3)

            override fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double {
                var total = 0.0
                point.indices.forEach { i ->
                    total += scales[i] * point[i] * point[i]
                    gradient[i] = 2.0 * scales[i] * point[i]
                }
                return total
            }
        }

        val result = LbfgsMinimizer(gradientTolerance = 1e-9, maxIterations = 1000)
            .minimize(function = illScaled, start = doubleArrayOf(500.0, 500.0, 500.0))

        assertTrue(actual = result.converged, message = "stopped after ${result.iterations} iterations")
        assertTrue(actual = result.point.all { entry -> abs(x = entry) < 1e-5 })
    }

    @Test
    internal fun `rejects a starting vector of the wrong size`() {
        assertFailsWith<IllegalArgumentException> {
            LbfgsMinimizer().minimize(
                function = Quadratic(centre = doubleArrayOf(0.0, 0.0)),
                start = doubleArrayOf(1.0),
            )
        }
    }

    @Test
    internal fun `rejects a non-positive configuration`() {
        assertFailsWith<IllegalArgumentException> { LbfgsMinimizer(historySize = 0) }
        assertFailsWith<IllegalArgumentException> { LbfgsMinimizer(maxIterations = 0) }
        assertFailsWith<IllegalArgumentException> { LbfgsMinimizer(gradientTolerance = 0.0) }
    }

    @Test
    internal fun `a history size of one still converges`() {
        val result = LbfgsMinimizer(historySize = 1, gradientTolerance = 1e-10, maxIterations = 5000)
            .minimize(function = Rosenbrock(), start = doubleArrayOf(-1.2, 1.0))

        assertTrue(actual = result.converged, message = "stopped after ${result.iterations} iterations")
    }
}
