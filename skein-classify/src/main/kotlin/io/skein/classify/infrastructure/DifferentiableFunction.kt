package io.skein.classify.infrastructure

/**
 * A scalar function of a vector that can report its own gradient — everything [LbfgsMinimizer]
 * needs to know about what it is minimising.
 *
 * Value and gradient are produced together rather than by two calls because every objective worth
 * minimising shares almost all of the work between them: for a logistic loss, one pass over the
 * design matrix yields both, and computing them separately doubles the cost of training.
 */
interface DifferentiableFunction {

    /** Length of the point and gradient vectors. */
    val dimension: Int

    /**
     * Evaluates the function at [point], writing the gradient into [gradient] and returning the
     * value.
     *
     * [gradient] is supplied by the caller and overwritten in full, so a minimiser can reuse one
     * array across every iteration instead of allocating a feature-sized `DoubleArray` per step.
     * Implementations must not retain either array.
     */
    fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double
}
