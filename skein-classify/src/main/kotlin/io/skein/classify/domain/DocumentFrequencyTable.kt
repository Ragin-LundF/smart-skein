package io.skein.classify.domain

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.roundToInt

/** Proportion of the corpus a bucket must reach to survive, once the corpus is large. */
private const val DOCUMENT_FREQUENCY_PROPORTION = 0.00002

/** Smallest usable floor: a bucket seen in one document alone can teach nothing. */
private const val MINIMUM_DOCUMENT_FREQUENCY_FLOOR = 2

/**
 * How many documents each feature bucket occurred in, and the inverse-document-frequency weight
 * that follows from it.
 *
 * Down-weights buckets that appear everywhere — with character n-grams, a great many do — so a
 * feature earns influence by being *discriminating* rather than by being common. The weighting is
 * scikit-learn's smoothed form, `ln((1 + n) / (1 + df)) + 1`, so a bucket in every document still
 * contributes rather than vanishing.
 *
 * **This table is fitted, and that makes it leakable.** Fit it on a fold's training rows alone. A
 * table fitted over the whole corpus carries the held-out rows' statistics into every training
 * feature, and every score afterwards is inflated —
 * [io.skein.classify.application.MultiLabelCrossValidator] is built to make that the easy path.
 *
 * [minimumDocumentFrequency] drops buckets seen too rarely to mean anything. Scale it with the
 * corpus via [floorFor]: a floor of 2 is right at seven thousand documents and absurd at a million,
 * where a bucket in two documents is indistinguishable from noise.
 */
class DocumentFrequencyTable(
    val documentCount: Int,
    val frequencies: IntArray,
    val minimumDocumentFrequency: Int,
) {

    init {
        require(value = documentCount > 0) { "documentCount must be positive, got $documentCount" }
        require(value = frequencies.isNotEmpty()) { "frequencies must cover at least one bucket" }
        require(value = minimumDocumentFrequency >= 1) { "minimumDocumentFrequency must be at least 1" }
    }

    /** Whether [bucket] reached the floor. A bucket below it contributes nothing to any vector. */
    fun isKept(bucket: Int): Boolean {
        if (bucket < 0 || bucket >= frequencies.size) {
            return false
        }
        return frequencies[bucket] >= minimumDocumentFrequency
    }

    /** The IDF weight of [bucket], or `0.0` when it fell below the floor. */
    fun idf(bucket: Int): Double {
        if (!isKept(bucket = bucket)) {
            return 0.0
        }
        return ln(x = (1.0 + documentCount) / (1.0 + frequencies[bucket])) + 1.0
    }

    /** Buckets that reached the floor — the feature count actually in play. */
    fun keptBucketCount(): Int {
        return frequencies.count { frequency -> frequency >= minimumDocumentFrequency }
    }

    /**
     * A digest over everything that changes an emitted vector, for
     * [io.skein.classify.spi.Vectorizer.fingerprint]. Covers every count, so a table refitted on
     * different data cannot be mistaken for this one.
     */
    fun digest(): String {
        val buffer = ByteBuffer.allocate(Int.SIZE_BYTES * (frequencies.size + 2)).order(ByteOrder.LITTLE_ENDIAN)
        buffer.putInt(documentCount)
        buffer.putInt(minimumDocumentFrequency)
        frequencies.forEach { frequency -> buffer.putInt(frequency) }
        val digest = MessageDigest.getInstance("SHA-256").digest(buffer.array())
        return digest.joinToString(separator = "") { byte -> "%02x".format(byte) }
    }

    companion object {

        /**
         * The document-frequency floor for a corpus of [documentCount] rows.
         *
         * A fixed floor cannot serve both ends of the range this library targets. Two documents out
         * of seven thousand is a real if weak signal; two out of a million is indistinguishable from
         * a hash collision, and keeping it fills the feature space with noise that pruning then has
         * to remove again. Scaling by a proportion keeps the meaning of "too rare" constant.
         */
        fun floorFor(documentCount: Int): Int {
            val proportional = (documentCount * DOCUMENT_FREQUENCY_PROPORTION).roundToInt()
            return max(a = MINIMUM_DOCUMENT_FREQUENCY_FLOOR, b = proportional)
        }
    }
}
