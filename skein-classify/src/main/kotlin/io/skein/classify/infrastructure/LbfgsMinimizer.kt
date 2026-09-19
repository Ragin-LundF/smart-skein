package io.skein.classify.infrastructure

import kotlin.math.sqrt

/** Curvature pairs retained. Memory is `historySize x 2 x dimension` doubles, per minimiser. */
private const val DEFAULT_HISTORY_SIZE = 5

/** Iterations before the minimiser gives up and reports the best point it reached. */
private const val DEFAULT_MAX_ITERATIONS = 500

/** Convergence test: the largest absolute gradient component must fall to this. */
private const val DEFAULT_GRADIENT_TOLERANCE = 1e-6

/** Armijo sufficient-decrease constant. The conventional value; the search is insensitive to it. */
private const val ARMIJO_CONSTANT = 1e-4

/** Step shrink factor per backtrack. */
private const val BACKTRACK_FACTOR = 0.5

/** Backtracking below this step length means the direction is useless; stop rather than crawl. */
private const val MINIMUM_STEP = 1e-20

/**
 * Limited-memory BFGS with an Armijo backtracking line search.
 *
 * A quasi-Newton method: it never forms or inverts a Hessian, but reconstructs the action of an
 * approximate inverse Hessian from the last [historySize] `(step, gradient change)` pairs via the
 * standard two-loop recursion. That is what makes it usable on a hundred thousand features, where
 * an explicit Hessian would need 10^10 doubles.
 *
 * **Complementary to online SGD, not a replacement.** This needs the whole objective at every step,
 * so it cannot learn from one observation at a time — see [io.skein.classify.spi.BatchLearner]. What
 * it offers in exchange is the true optimum: measured on the reference problem it reaches a gradient
 * norm around 1e-11, where a stochastic method is still wandering and where scikit-learn's default
 * `tol = 1e-4` stops roughly 4e-4 short.
 *
 * Memory is `historySize x 2 x dimension` doubles **per instance**, which is the trap when fitting
 * many labels in parallel: at 370,000 features one minimiser holds 59 MB, and eighteen of them hold
 * a gigabyte. Bound the parallelism from the heap, not from the core count —
 * [TrainingParallelism] does exactly that.
 */
class LbfgsMinimizer(
    private val historySize: Int = DEFAULT_HISTORY_SIZE,
    private val maxIterations: Int = DEFAULT_MAX_ITERATIONS,
    private val gradientTolerance: Double = DEFAULT_GRADIENT_TOLERANCE,
) {

    init {
        require(value = historySize > 0) { "historySize must be positive, got $historySize" }
        require(value = maxIterations > 0) { "maxIterations must be positive, got $maxIterations" }
        require(value = gradientTolerance > 0.0) { "gradientTolerance must be positive" }
    }

    /**
     * Minimises [function], starting from [start] or from the origin when it is `null`.
     *
     * [start] is copied, never mutated, so a caller may reuse a warm-start vector across labels.
     */
    fun minimize(function: DifferentiableFunction, start: DoubleArray? = null): LbfgsResult {
        val dimension = function.dimension
        require(value = start == null || start.size == dimension) {
            "start has ${start?.size} entries but the function has $dimension"
        }
        val point = start?.copyOf() ?: DoubleArray(size = dimension)
        val gradient = DoubleArray(size = dimension)
        var value = function.valueAndGradient(point = point, gradient = gradient)

        val history = LbfgsHistory(dimension = dimension, capacity = historySize)
        val direction = DoubleArray(size = dimension)
        val trialPoint = DoubleArray(size = dimension)
        val trialGradient = DoubleArray(size = dimension)

        var iterations = 0
        while (iterations < maxIterations) {
            if (VectorMath.maximumAbsolute(vector = gradient) <= gradientTolerance) {
                return result(point = point, value = value, gradient = gradient, iterations = iterations, done = true)
            }
            history.descentDirection(gradient = gradient, into = direction)
            val slope = VectorMath.dot(left = gradient, right = direction)
            val initialStep = initialStep(history = history, gradient = gradient)
            val accepted = lineSearch(
                function = function,
                point = point,
                direction = direction,
                value = value,
                slope = slope,
                initialStep = initialStep,
                trialPoint = trialPoint,
                trialGradient = trialGradient,
            )
            if (accepted == null) {
                // The direction yields no decrease at any usable step length. Further iterations
                // would repeat the same search, so report the best point rather than spin.
                return result(point = point, value = value, gradient = gradient, iterations = iterations, done = false)
            }
            history.record(
                previousPoint = point,
                previousGradient = gradient,
                point = trialPoint,
                gradient = trialGradient,
            )
            trialPoint.copyInto(destination = point)
            trialGradient.copyInto(destination = gradient)
            // The line search already evaluated this exact point, and [trialGradient] holds its
            // gradient. Re-evaluating here would double the cost of every iteration -- the
            // objective is a full pass over the design matrix, which is essentially the whole cost
            // of training.
            value = accepted
            iterations++
        }
        return result(point = point, value = value, gradient = gradient, iterations = iterations, done = false)
    }

    /**
     * Armijo backtracking: the smallest number of halvings that buys a decrease proportional to the
     * directional derivative. Returns the objective **value** at the accepted point, or `null` when
     * no usable step was found.
     *
     * Returning the value rather than the step is what lets the caller avoid re-evaluating: on
     * success [trialPoint] and [trialGradient] already hold the accepted point and its gradient, so
     * one iteration costs one objective evaluation plus one per backtrack, not two.
     */
    @Suppress("LongParameterList")
    private fun lineSearch(
        function: DifferentiableFunction,
        point: DoubleArray,
        direction: DoubleArray,
        value: Double,
        slope: Double,
        initialStep: Double,
        trialPoint: DoubleArray,
        trialGradient: DoubleArray,
    ): Double? {
        var step = initialStep
        while (step >= MINIMUM_STEP) {
            point.indices.forEach { i -> trialPoint[i] = point[i] + step * direction[i] }
            val trialValue = function.valueAndGradient(point = trialPoint, gradient = trialGradient)
            // A non-finite trial is a step into an overflow, not a candidate; shrink and retry.
            if (trialValue.isFinite() && trialValue <= value + ARMIJO_CONSTANT * step * slope) {
                return trialValue
            }
            step *= BACKTRACK_FACTOR
        }
        return null
    }

    /**
     * The first step is scaled by the inverse gradient norm because the initial search direction is
     * the raw steepest descent, whose length carries the gradient's scale rather than the problem's.
     * A unit step along it overshoots by orders of magnitude on a steep problem. Once curvature
     * pairs exist the approximate inverse Hessian supplies the scale, so a unit step is right.
     */
    private fun initialStep(history: LbfgsHistory, gradient: DoubleArray): Double {
        if (history.isNotEmpty()) {
            return 1.0
        }
        val norm = sqrt(x = VectorMath.dot(left = gradient, right = gradient))
        return if (norm > 0.0) 1.0 / norm else 1.0
    }

    private fun result(
        point: DoubleArray,
        value: Double,
        gradient: DoubleArray,
        iterations: Int,
        done: Boolean,
    ): LbfgsResult {
        return LbfgsResult(
            point = point,
            value = value,
            gradientNorm = VectorMath.maximumAbsolute(vector = gradient),
            iterations = iterations,
            converged = done,
        )
    }

}
