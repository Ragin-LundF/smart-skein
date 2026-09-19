package io.skein.classify.infrastructure

import kotlin.math.abs

/**
 * Dense vector primitives shared by the L-BFGS machinery.
 *
 * Hand-written loops rather than stream or `zip` idioms: these run once per feature per iteration
 * per label, so an intermediate allocation here is multiplied by hundreds of millions.
 */
internal object VectorMath {

    /** The inner product of two equally-sized vectors. */
    fun dot(left: DoubleArray, right: DoubleArray): Double {
        var total = 0.0
        left.indices.forEach { i -> total += left[i] * right[i] }
        return total
    }

    /** The infinity norm — the convergence test L-BFGS is conventionally stopped on. */
    fun maximumAbsolute(vector: DoubleArray): Double {
        var largest = 0.0
        vector.forEach { entry -> largest = maxOf(a = largest, b = abs(x = entry)) }
        return largest
    }
}
