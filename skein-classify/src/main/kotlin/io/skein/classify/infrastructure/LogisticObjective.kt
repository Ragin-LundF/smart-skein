package io.skein.classify.infrastructure

import io.skein.classify.domain.MultiLabelPredictionFactory
import io.skein.classify.domain.SparseMatrix
import java.util.Arrays
import kotlin.math.exp
import kotlin.math.ln

/** The `1/2` in `0.5 * w.w`, which is what makes the penalty's derivative a plain `w`. */
private const val L2_PENALTY_COEFFICIENT = 0.5

/**
 * The L2-regularised binary logistic loss of one one-vs-rest head over a [SparseMatrix], in the
 * form [LbfgsMinimizer] can minimise.
 *
 * The objective is scikit-learn's, exactly:
 *
 * ```
 * f(w, b) = 0.5 * w.w + C * sum_i softplus(-y_i * (x_i.w + b)),   y_i in {-1, +1}
 * ```
 *
 * Two conventions in there are load-bearing, and getting either wrong produces a model that trains
 * happily and scores differently from every reference:
 * - **The intercept is not regularised.** It absorbs the class prior, and shrinking it towards zero
 *   biases a rare label's predictions towards the majority. It lives at the last index of the
 *   parameter vector and is excluded from the penalty.
 * - **`C` multiplies the data term, not the penalty.** That is what makes a `C` here mean the same
 *   thing as a `C` in scikit-learn, so their solutions are comparable rather than merely similar.
 *
 * Positives are supplied as sorted row indices rather than a per-row flag. A `BooleanArray` per
 * label would cost one byte per document per label — 237 MB across the reference taxonomy at a
 * million documents — where the index list costs only as much as the label's actual support.
 */
class LogisticObjective(
    private val matrix: SparseMatrix,
    private val positiveRows: IntArray,
    private val inverseRegularization: Double,
) : DifferentiableFunction {

    init {
        require(value = inverseRegularization > 0.0) {
            "inverseRegularization (C) must be positive, got $inverseRegularization"
        }
        require(value = positiveRows.all { row -> row in 0 until matrix.rowCount }) {
            "every positive row must be within 0 until ${matrix.rowCount}"
        }
        require(value = isStrictlyAscending(rows = positiveRows)) {
            "positiveRows must be sorted ascending and free of duplicates"
        }
    }

    /** One weight per feature, plus the intercept at the last index. */
    override val dimension: Int = matrix.columnCount + 1

    private val interceptIndex: Int = matrix.columnCount

    override fun valueAndGradient(point: DoubleArray, gradient: DoubleArray): Double {
        require(value = point.size == dimension) { "point has ${point.size} entries but expected $dimension" }
        Arrays.fill(gradient, 0.0)
        val intercept = point[interceptIndex]
        var loss = 0.0
        var interceptGradient = 0.0
        var cursor = 0

        matrix.blocks.forEach { block ->
            val offsets = block.rowOffsets
            val columns = block.columnIndices
            val values = block.values
            for (local in 0 until block.rowCount()) {
                val row = block.firstRow + local
                while (cursor < positiveRows.size && positiveRows[cursor] < row) {
                    cursor++
                }
                val sign = if (cursor < positiveRows.size && positiveRows[cursor] == row) 1.0 else -1.0
                val from = offsets[local]
                val to = offsets[local + 1]
                var score = intercept
                for (k in from until to) {
                    score += point[columns[k]] * values[k]
                }
                val margin = -sign * score
                loss += softplus(value = margin)
                val slope = -sign * MultiLabelPredictionFactory.logistic(logit = margin)
                for (k in from until to) {
                    gradient[columns[k]] += slope * values[k]
                }
                interceptGradient += slope
            }
        }

        var penalty = 0.0
        for (j in 0 until interceptIndex) {
            val weight = point[j]
            penalty += weight * weight
            gradient[j] = gradient[j] * inverseRegularization + weight
        }
        gradient[interceptIndex] = interceptGradient * inverseRegularization
        return L2_PENALTY_COEFFICIENT * penalty + inverseRegularization * loss
    }

    /**
     * `ln(1 + exp(value))`, evaluated on whichever branch cannot overflow.
     *
     * The naive form returns infinity from `exp` for any argument past ~710, and a margin of
     * several hundred is ordinary once a few hundred agreeing features are summed. A single
     * infinite loss makes the whole objective infinite, the line search rejects every step, and the
     * fit stops at the starting point — a failure that looks like non-convergence rather than
     * overflow.
     */
    private fun softplus(value: Double): Double {
        if (value > 0.0) {
            return value + ln(x = 1.0 + exp(x = -value))
        }
        return ln(x = 1.0 + exp(x = value))
    }

    private fun isStrictlyAscending(rows: IntArray): Boolean {
        for (i in 1 until rows.size) {
            if (rows[i] <= rows[i - 1]) {
                return false
            }
        }
        return true
    }
}
