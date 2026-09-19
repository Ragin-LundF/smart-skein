package io.skein.classify.application

import io.skein.classify.domain.DocumentFrequencyTable
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.spi.Vectorizer
import java.security.MessageDigest

/**
 * Weights another vectorizer's features by inverse document frequency, and drops buckets that are
 * too rare to mean anything.
 *
 * A decorator rather than a replacement: it wraps [HashingVectorizer] or any other [Vectorizer],
 * keeps that vectorizer's feature space exactly, and only rescales the values. Nothing downstream
 * changes — the learner, the objective and the scoring loop never learn that IDF happened.
 *
 * **An optional refinement, and one to justify by measurement.** It costs a fitted table that must
 * travel with the model and be refitted per cross-validation fold, and on a corpus whose records
 * are short and similar in length it often buys very little. Measure against plain counts before
 * adopting it.
 *
 * Its [fingerprint] composes the delegate's with a digest of the whole table, so a model trained
 * against one fitted table cannot be scored against another — which matters more here than for
 * plain hashing, because a refit changes every value in the vector while leaving every index alone.
 */
class IdfVectorizer(
    private val delegate: Vectorizer,
    private val table: DocumentFrequencyTable,
) : Vectorizer {

    override fun vectorize(text: String): FeatureVector {
        val raw = delegate.vectorize(text = text)
        val indices = IntArray(size = raw.indices.size)
        val values = FloatArray(size = raw.indices.size)
        var kept = 0
        for (position in raw.indices.indices) {
            val bucket = raw.indices[position]
            val weight = table.idf(bucket = bucket)
            if (weight == 0.0) {
                continue
            }
            indices[kept] = bucket
            values[kept] = (raw.values[position] * weight).toFloat()
            kept++
        }
        return FeatureVector(
            indices = indices.copyOf(newSize = kept),
            values = values.copyOf(newSize = kept),
        )
    }

    override fun dimension(): Int {
        return delegate.dimension()
    }

    override fun fingerprint(): VectorizerFingerprint {
        val inner = delegate.fingerprint()
        val material = "${inner.kind}|${inner.dimension}|${inner.configDigest}|${table.digest()}"
        val digest = MessageDigest.getInstance("SHA-256").digest(material.toByteArray(Charsets.UTF_8))
        return VectorizerFingerprint(
            kind = "idf(${inner.kind})",
            dimension = inner.dimension,
            configDigest = digest.joinToString(separator = "") { byte -> "%02x".format(byte) },
        )
    }

    /** The fitted table, so a caller can persist it alongside the model. */
    fun table(): DocumentFrequencyTable {
        return table
    }

    companion object {

        /**
         * Fits a table over [texts] using [delegate]'s feature space, then returns the wrapped
         * vectorizer.
         *
         * Pass a fold's **training rows only**. The floor defaults to
         * [DocumentFrequencyTable.floorFor] of the corpus size, which scales with it.
         */
        fun fit(
            delegate: Vectorizer,
            texts: List<String>,
            minimumDocumentFrequency: Int = DocumentFrequencyTable.floorFor(documentCount = texts.size),
        ): IdfVectorizer {
            require(value = texts.isNotEmpty()) { "cannot fit document frequencies on an empty corpus" }
            val frequencies = IntArray(size = delegate.dimension())
            texts.forEach { text ->
                // distinct(): document frequency counts documents, not occurrences, so a bucket hit
                // twice in one record must still only advance its count by one.
                delegate.vectorize(text = text).indices.distinct().forEach { bucket ->
                    frequencies[bucket]++
                }
            }
            return IdfVectorizer(
                delegate = delegate,
                table = DocumentFrequencyTable(
                    documentCount = texts.size,
                    frequencies = frequencies,
                    minimumDocumentFrequency = minimumDocumentFrequency,
                ),
            )
        }
    }
}
