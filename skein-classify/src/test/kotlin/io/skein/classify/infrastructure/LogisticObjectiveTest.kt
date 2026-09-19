package io.skein.classify.infrastructure

import io.skein.classify.domain.SparseMatrixBuilder
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class LogisticObjectiveTest {

    private val problem = LogisticReferenceProblem.load()

    /**
     * Verifies the analytic gradient against central finite differences.
     *
     * This is the test that catches a wrong derivation, and it does so **without restating the
     * derivation**: the numeric slope is built only from values of the objective, so if the
     * hand-written gradient disagrees with the function it belongs to, the two diverge. A test that
     * recomputed the gradient formula would agree with a wrong formula.
     */
    @Test
    internal fun `analytic gradient matches central finite differences`() {
        val objective = problem.objective()
        val point = doubleArrayOf(0.4, -0.7, 1.1, 0.3)
        val analytic = DoubleArray(size = objective.dimension)
        objective.valueAndGradient(point = point, gradient = analytic)

        val step = 1e-6
        val scratch = DoubleArray(size = objective.dimension)
        point.indices.forEach { j ->
            val probe = point.copyOf()
            probe[j] = point[j] + step
            val forward = objective.valueAndGradient(point = probe, gradient = scratch)
            probe[j] = point[j] - step
            val backward = objective.valueAndGradient(point = probe, gradient = scratch)
            val numeric = (forward - backward) / (2.0 * step)
            assertEquals(
                expected = numeric,
                actual = analytic[j],
                absoluteTolerance = 1e-6,
                message = "gradient component $j disagrees with the numeric slope",
            )
        }
    }

    @Test
    internal fun `stays finite at margins that overflow a naive exponential`() {
        val objective = problem.objective()
        val gradient = DoubleArray(size = objective.dimension)

        // Weights this large push every row's margin past +-800, where exp() is Infinity.
        val extreme = doubleArrayOf(0.0, 0.0, 0.0, 800.0)
        val value = objective.valueAndGradient(point = extreme, gradient = gradient)

        assertTrue(actual = value.isFinite(), message = "objective overflowed at a margin of 800")
        assertTrue(actual = gradient.all { entry -> entry.isFinite() }, message = "gradient overflowed")
        assertTrue(
            actual = objective.valueAndGradient(point = doubleArrayOf(0.0, 0.0, 0.0, -800.0), gradient = gradient)
                .isFinite(),
        )
    }

    /**
     * A one-column matrix whose only feature is always 1.0 makes the feature weight and the
     * intercept enter the score identically: `z = w * 1 + b`. Any difference between moving `w`
     * and moving `b` is therefore the penalty and nothing else, which pins the convention
     * directly rather than by inference.
     */
    private fun interceptEquivalentObjective(): LogisticObjective {
        val builder = SparseMatrixBuilder(columnCount = 1)
        repeat(times = 6) {
            builder.addRow(columnIndices = intArrayOf(0), values = floatArrayOf(1.0f), from = 0, to = 1)
        }
        return LogisticObjective(
            matrix = builder.build(),
            positiveRows = intArrayOf(0, 2, 4),
            inverseRegularization = 1.0,
        )
    }

    @Test
    internal fun `the intercept is not regularised`() {
        val objective = interceptEquivalentObjective()
        val gradient = DoubleArray(size = 2)
        val weight = 0.75

        val asFeature = objective.valueAndGradient(point = doubleArrayOf(weight, 0.0), gradient = gradient)
        val asIntercept = objective.valueAndGradient(point = doubleArrayOf(0.0, weight), gradient = gradient)

        // Identical data term, so the whole difference is the feature weight's penalty.
        assertEquals(
            expected = 0.5 * weight * weight,
            actual = asFeature - asIntercept,
            absoluteTolerance = 1e-12,
        )
    }

    @Test
    internal fun `only the feature gradient carries the penalty derivative`() {
        val objective = interceptEquivalentObjective()
        val gradient = DoubleArray(size = 2)
        val weight = 0.75

        objective.valueAndGradient(point = doubleArrayOf(weight, 0.0), gradient = gradient)
        val featureGradient = gradient[0]
        objective.valueAndGradient(point = doubleArrayOf(0.0, weight), gradient = gradient)
        val interceptGradient = gradient[1]

        // d/dw adds the penalty's `+ w`; d/db adds nothing.
        assertEquals(expected = weight, actual = featureGradient - interceptGradient, absoluteTolerance = 1e-12)
    }

    @Test
    internal fun `reports the reference objective value at the reference optimum`() {
        val objective = problem.objective()
        val gradient = DoubleArray(size = objective.dimension)

        val value = objective.valueAndGradient(point = problem.optimum.toDoubleArray(), gradient = gradient)

        assertEquals(expected = problem.optimalValue, actual = value, absoluteTolerance = 1e-12)
    }

    @Test
    internal fun `the gradient vanishes at the reference optimum`() {
        val objective = problem.objective()
        val gradient = DoubleArray(size = objective.dimension)

        objective.valueAndGradient(point = problem.optimum.toDoubleArray(), gradient = gradient)

        assertTrue(
            actual = gradient.all { entry -> abs(x = entry) < 1e-12 },
            message = "gradient at the Newton optimum was ${gradient.toList()}",
        )
    }

    @Test
    internal fun `C scales the data term without touching the penalty`() {
        val matrix = problem.matrix()
        val gradient = DoubleArray(size = matrix.columnCount + 1)
        val point = doubleArrayOf(0.3, 0.0, 0.0, 0.0)

        val weak = LogisticObjective(
            matrix = matrix,
            positiveRows = problem.positiveRows(),
            inverseRegularization = 1.0,
        ).valueAndGradient(point = point, gradient = gradient)
        val strong = LogisticObjective(
            matrix = matrix,
            positiveRows = problem.positiveRows(),
            inverseRegularization = 2.0,
        ).valueAndGradient(point = point, gradient = gradient)

        // penalty = 0.5 * 0.3^2 = 0.045, identical in both; the remainder doubles.
        val penalty = 0.5 * 0.3 * 0.3
        assertEquals(expected = 2.0 * (weak - penalty) + penalty, actual = strong, absoluteTolerance = 1e-12)
    }

    @Test
    internal fun `an all-negative corpus drives the intercept gradient negative`() {
        val builder = SparseMatrixBuilder(columnCount = 1)
        repeat(times = 4) {
            builder.addRow(columnIndices = intArrayOf(0), values = floatArrayOf(1.0f), from = 0, to = 1)
        }
        val objective = LogisticObjective(
            matrix = builder.build(),
            positiveRows = intArrayOf(),
            inverseRegularization = 1.0,
        )
        val gradient = DoubleArray(size = 2)

        objective.valueAndGradient(point = doubleArrayOf(0.0, 0.0), gradient = gradient)

        // Every row is negative, so raising the intercept raises the loss: the gradient is positive.
        assertTrue(actual = gradient[1] > 0.0, message = "expected a positive intercept gradient")
    }

    @Test
    internal fun `rejects a non-positive regularisation strength`() {
        assertFailsWith<IllegalArgumentException> {
            LogisticObjective(matrix = problem.matrix(), positiveRows = intArrayOf(0), inverseRegularization = 0.0)
        }
    }

    @Test
    internal fun `rejects positive rows that are unsorted or out of range`() {
        assertFailsWith<IllegalArgumentException> {
            LogisticObjective(
                matrix = problem.matrix(),
                positiveRows = intArrayOf(2, 1),
                inverseRegularization = 1.0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            LogisticObjective(
                matrix = problem.matrix(),
                positiveRows = intArrayOf(1, 1),
                inverseRegularization = 1.0,
            )
        }
        assertFailsWith<IllegalArgumentException> {
            LogisticObjective(
                matrix = problem.matrix(),
                positiveRows = intArrayOf(9999),
                inverseRegularization = 1.0,
            )
        }
    }

    @Test
    internal fun `rejects a point of the wrong size`() {
        val objective = problem.objective()

        assertFailsWith<IllegalArgumentException> {
            objective.valueAndGradient(point = doubleArrayOf(1.0), gradient = DoubleArray(size = 4))
        }
    }
}
