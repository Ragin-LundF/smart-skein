package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.roundToLong

/**
 * The fitted weights of a one-vs-rest linear model, stored **feature-major**.
 *
 * Feature-major is the whole performance story. A record activates a couple of hundred features out
 * of a quarter of a million, so scoring walks those rows and accumulates into a labels-sized array,
 * touching only the weights that could possibly matter. The label-major alternative — one weight
 * vector per label — forces a scan of every label for every record, which is two orders of
 * magnitude more work at the same accuracy. It is most of why the reference implementation reaches
 * 24,000 records/s on one core.
 *
 * Layout is CSR over features: feature `f` owns
 * `labelIndices[featureOffsets[f] until featureOffsets[f + 1]]`, ascending, with [weights] aligned
 * by position. [intercepts] is dense, one per label, and is never pruned — it carries the class
 * prior, and dropping it would bias every rare label towards the majority.
 *
 * Immutable, so one instance is safe to score from any number of threads.
 */
class MultiLabelWeights(
    val labels: List<Label>,
    val intercepts: DoubleArray,
    val featureOffsets: IntArray,
    val labelIndices: IntArray,
    val weights: FloatArray,
) {

    init {
        require(value = labels.isNotEmpty()) { "a fitted model must know at least one label" }
        require(value = labels.size == intercepts.size) { "one intercept per label is required" }
        require(value = featureOffsets.isNotEmpty()) { "featureOffsets must carry at least the terminator" }
        require(value = labelIndices.size == weights.size) { "labelIndices and weights must have equal length" }
        require(value = featureOffsets.last() == weights.size) {
            "featureOffsets terminator ${featureOffsets.last()} must equal the weight count ${weights.size}"
        }
    }

    /** Width of the feature space this model was fitted over. */
    val featureCount: Int = featureOffsets.size - 1

    /** Weights actually stored, after pruning. */
    fun nonZeroCount(): Int {
        return weights.size
    }

    /**
     * Scores [features] against every label at once, returning raw logits indexed as [labels].
     *
     * Features outside this model's space are skipped rather than rejected: a vectorizer configured
     * with more buckets than the model was fitted over is a configuration error caught by the
     * fingerprint check on load, and failing here instead would turn it into a per-record exception.
     */
    fun logits(features: FeatureVector): DoubleArray {
        val scores = intercepts.copyOf()
        for (position in features.indices.indices) {
            val feature = features.indices[position]
            if (feature < 0 || feature >= featureCount) {
                continue
            }
            val value = features.values[position].toDouble()
            val from = featureOffsets[feature]
            val to = featureOffsets[feature + 1]
            for (k in from until to) {
                scores[labelIndices[k]] += weights[k] * value
            }
        }
        return scores
    }

    /** The weight of [feature] for the label at [labelIndex], or `0.0` when it was pruned away. */
    fun weightOf(feature: Int, labelIndex: Int): Double {
        if (feature < 0 || feature >= featureCount) {
            return 0.0
        }
        val from = featureOffsets[feature]
        val to = featureOffsets[feature + 1]
        val found = Arrays.binarySearch(labelIndices, from, to, labelIndex)
        return if (found >= 0) weights[found].toDouble() else 0.0
    }

    companion object {

        /**
         * Builds a pruned feature-major matrix from one fitted weight vector per label.
         *
         * **Why prune at all.** L2 shrinks weights towards zero but never to it, so a fitted matrix
         * is fully dense and overwhelmingly noise: the reference model is 24 million weights of
         * which 1.9 million carry the decisions. Keeping the largest [keepFraction] by magnitude
         * cut a 195 MB artefact to 17 MB while changing one label decision in 1.7 million.
         *
         * The cutoff is global rather than per label, which is deliberate and has a known cost: a
         * label with few examples has uniformly small weights and can lose most of them, while a
         * well-supported label keeps many. That tracks how much each label was actually learned,
         * but it does mean a rare label can be pruned into silence — see the per-label floor in the
         * model-size plan if the tail matters more than the bytes.
         *
         * Ties at the cutoff are kept, so perfectly correlated features survive together. On a
         * corpus large enough to matter this is a handful of extra weights; on a tiny one it can
         * keep noticeably more than [keepFraction].
         */
        fun fromLabelMajor(
            labels: List<Label>,
            intercepts: DoubleArray,
            weightsByLabel: List<FloatArray>,
            featureCount: Int,
            keepFraction: Double,
        ): MultiLabelWeights {
            require(value = keepFraction > 0.0 && keepFraction <= 1.0) {
                "keepFraction must be in (0, 1], got $keepFraction"
            }
            require(value = weightsByLabel.size == labels.size) { "one weight vector per label is required" }
            require(value = weightsByLabel.all { vector -> vector.size == featureCount }) {
                "every weight vector must hold $featureCount entries"
            }
            val cutoff = magnitudeCutoff(weightsByLabel = weightsByLabel, keepFraction = keepFraction)
            return pack(
                labels = labels,
                intercepts = intercepts,
                weightsByLabel = weightsByLabel,
                featureCount = featureCount,
                cutoff = cutoff,
            )
        }

        /**
         * The magnitude at or above which a weight is kept.
         *
         * ponytail: sorts a copy of every magnitude, which costs a `FloatArray` the size of the
         * dense matrix — 96 MB at the reference model's 24 million weights. Ceiling: a taxonomy an
         * order of magnitude larger makes the transient uncomfortable. Upgrade path: a two-pass
         * histogram select over magnitude buckets, which is O(1) in memory and needs no copy.
         */
        private fun magnitudeCutoff(weightsByLabel: List<FloatArray>, keepFraction: Double): Float {
            if (keepFraction >= 1.0) {
                return 0.0f
            }
            val total = weightsByLabel.sumOf { vector -> vector.size.toLong() }
            val keep = (total * keepFraction).roundToLong().coerceAtLeast(minimumValue = 1L)
            if (keep >= total) {
                return 0.0f
            }
            val magnitudes = FloatArray(size = total.toInt())
            var at = 0
            weightsByLabel.forEach { vector ->
                vector.forEach { weight ->
                    magnitudes[at++] = abs(x = weight)
                }
            }
            Arrays.sort(magnitudes)
            return magnitudes[(total - keep).toInt()]
        }

        private fun pack(
            labels: List<Label>,
            intercepts: DoubleArray,
            weightsByLabel: List<FloatArray>,
            featureCount: Int,
            cutoff: Float,
        ): MultiLabelWeights {
            val offsets = IntArray(size = featureCount + 1)
            for (feature in 0 until featureCount) {
                var kept = 0
                weightsByLabel.forEach { vector ->
                    if (abs(x = vector[feature]) >= cutoff && vector[feature] != 0.0f) {
                        kept++
                    }
                }
                offsets[feature + 1] = offsets[feature] + kept
            }
            val total = offsets[featureCount]
            val indices = IntArray(size = total)
            val values = FloatArray(size = total)
            var at = 0
            for (feature in 0 until featureCount) {
                weightsByLabel.forEachIndexed { labelIndex, vector ->
                    val weight = vector[feature]
                    if (abs(x = weight) >= cutoff && weight != 0.0f) {
                        indices[at] = labelIndex
                        values[at] = weight
                        at++
                    }
                }
            }
            return MultiLabelWeights(
                labels = labels,
                intercepts = intercepts,
                featureOffsets = offsets,
                labelIndices = indices,
                weights = values,
            )
        }
    }
}
