package io.skein.classify.infrastructure

/** A curvature pair is kept only when `s.y` exceeds this multiple of `y.y`. */
private const val CURVATURE_EPSILON = 1e-10

/**
 * The bounded curvature history behind [LbfgsMinimizer], and the two-loop recursion that turns it
 * into a search direction.
 *
 * Holds the last `capacity` pairs of `(step taken, resulting gradient change)`. Their inner products
 * are all the algorithm ever needs of the inverse Hessian, which is why L-BFGS runs on problems
 * where the Hessian itself would not fit in memory.
 *
 * Evicted arrays are recycled rather than dropped: a five-hundred-iteration fit over a hundred
 * thousand features would otherwise churn about 800 MB of short-lived `DoubleArray`s for nothing.
 */
internal class LbfgsHistory(private val dimension: Int, private val capacity: Int) {

    private val steps = ArrayDeque<DoubleArray>()
    private val gradientChanges = ArrayDeque<DoubleArray>()
    private val inverseCurvature = ArrayDeque<Double>()
    private val recycled = ArrayDeque<DoubleArray>()
    private val coefficients = DoubleArray(size = capacity)

    fun isNotEmpty(): Boolean {
        return steps.isNotEmpty()
    }

    /**
     * Records the pair produced by one accepted step, or discards it.
     *
     * A pair with non-positive curvature (`s.y <= 0`) would make the approximate inverse Hessian
     * indefinite and the next "descent" direction an ascent. Skipping such a pair is the standard
     * damping rule and is why the guard is an inequality rather than an assertion: on a
     * well-conditioned problem it never fires, and near a flat optimum it fires routinely.
     */
    fun record(
        previousPoint: DoubleArray,
        previousGradient: DoubleArray,
        point: DoubleArray,
        gradient: DoubleArray,
    ) {
        val step = borrow()
        val change = borrow()
        point.indices.forEach { i ->
            step[i] = point[i] - previousPoint[i]
            change[i] = gradient[i] - previousGradient[i]
        }
        val curvature = VectorMath.dot(left = step, right = change)
        val changeNorm = VectorMath.dot(left = change, right = change)
        if (curvature <= CURVATURE_EPSILON * changeNorm) {
            recycled.addLast(element = step)
            recycled.addLast(element = change)
            return
        }
        if (steps.size == capacity) {
            recycled.addLast(element = steps.removeFirst())
            recycled.addLast(element = gradientChanges.removeFirst())
            inverseCurvature.removeFirst()
        }
        steps.addLast(element = step)
        gradientChanges.addLast(element = change)
        inverseCurvature.addLast(element = 1.0 / curvature)
    }

    /**
     * Writes the L-BFGS search direction for [gradient] into [into] via the two-loop recursion.
     *
     * With no history this is plain steepest descent. The middle scaling by
     * `(s.y) / (y.y)` of the newest pair is the standard initial-Hessian estimate: without it the
     * first quasi-Newton step is badly scaled and the line search wastes most of its budget
     * backtracking.
     *
     * Falls back to steepest descent if the result is not a descent direction, which rounding can
     * produce very close to an optimum. Without the guard the line search would search uphill,
     * fail, and end the fit just short of the minimum.
     */
    fun descentDirection(gradient: DoubleArray, into: DoubleArray) {
        gradient.copyInto(destination = into)
        for (i in steps.indices.reversed()) {
            val coefficient = inverseCurvature[i] * VectorMath.dot(left = steps[i], right = into)
            coefficients[i] = coefficient
            val change = gradientChanges[i]
            into.indices.forEach { j -> into[j] -= coefficient * change[j] }
        }
        val scale = initialScale()
        into.indices.forEach { j -> into[j] *= scale }
        for (i in steps.indices) {
            val beta = inverseCurvature[i] * VectorMath.dot(left = gradientChanges[i], right = into)
            val adjustment = coefficients[i] - beta
            val step = steps[i]
            into.indices.forEach { j -> into[j] += adjustment * step[j] }
        }
        into.indices.forEach { j -> into[j] = -into[j] }
        if (VectorMath.dot(left = gradient, right = into) >= 0.0) {
            gradient.indices.forEach { j -> into[j] = -gradient[j] }
        }
    }

    private fun initialScale(): Double {
        if (steps.isEmpty()) {
            return 1.0
        }
        val newest = steps.size - 1
        val change = gradientChanges[newest]
        val changeNorm = VectorMath.dot(left = change, right = change)
        if (changeNorm == 0.0) {
            return 1.0
        }
        return VectorMath.dot(left = steps[newest], right = change) / changeNorm
    }

    private fun borrow(): DoubleArray {
        return recycled.removeLastOrNull() ?: DoubleArray(size = dimension)
    }
}
