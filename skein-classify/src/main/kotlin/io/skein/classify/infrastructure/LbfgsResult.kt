package io.skein.classify.infrastructure

/**
 * What [LbfgsMinimizer.minimize] reached.
 *
 * [converged] distinguishes "the gradient fell below the tolerance" from "the iteration or step
 * budget ran out". A caller that cares about optimality must check it: an unconverged result is
 * still the best point found, and is often perfectly usable, but it is not a minimum.
 *
 * Deliberately a plain class rather than a `data class`: [point] is an array, and generated
 * `equals`/`hashCode` over an array compare identity, which is a trap in a value-looking type.
 */
class LbfgsResult(
    val point: DoubleArray,
    val value: Double,
    val gradientNorm: Double,
    val iterations: Int,
    val converged: Boolean,
)
