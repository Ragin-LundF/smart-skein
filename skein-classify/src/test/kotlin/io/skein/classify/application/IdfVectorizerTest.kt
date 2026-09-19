package io.skein.classify.application

import io.skein.classify.domain.DocumentFrequencyTable
import io.skein.classify.domain.HashingConfig
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class IdfVectorizerTest {

    private val base = HashingVectorizer(config = HashingConfig(key0 = 3L, key1 = 5L, numFeatures = 1 shl 12))

    private val corpus = listOf(
        "monthly rent payment for the apartment",
        "monthly rent payment for the studio",
        "monthly rent payment for the loft",
        "unique zzqx wording nobody repeats",
    )

    @Test
    internal fun `down-weights a bucket that occurs in every document`() {
        val vectorizer = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)
        val everywhere = base.vectorize(text = "monthly").indices.first()
        val rare = base.vectorize(text = "zzqx").indices.first()

        assertTrue(
            actual = vectorizer.table().idf(bucket = rare) > vectorizer.table().idf(bucket = everywhere),
            message = "a rare bucket must outweigh a ubiquitous one",
        )
    }

    @Test
    internal fun `drops a bucket below the document-frequency floor`() {
        // "zzqx" appears in exactly one document, and its character n-grams appear nowhere else.
        // Most other buckets in that row do occur elsewhere -- three- to five-character n-grams of
        // ordinary English overlap across unrelated sentences -- so this checks the one bucket that
        // genuinely is unique rather than the whole row.
        val unique = base.vectorize(text = "zzqx").indices.toSet()

        val permissive = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)
        val strict = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 3)
        val row = "unique zzqx wording nobody repeats"

        assertTrue(actual = permissive.vectorize(text = row).indices.any { bucket -> bucket in unique })
        assertTrue(actual = strict.vectorize(text = row).indices.none { bucket -> bucket in unique })
    }

    @Test
    internal fun `raising the floor keeps strictly fewer buckets`() {
        val permissive = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)
        val strict = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 3)

        assertTrue(
            actual = strict.table().keptBucketCount() < permissive.table().keptBucketCount(),
            message = "${strict.table().keptBucketCount()} was not below ${permissive.table().keptBucketCount()}",
        )
    }

    @Test
    internal fun `keeps the delegate's feature space and index assignment`() {
        val vectorizer = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)

        val raw = base.vectorize(text = corpus.first())
        val weighted = vectorizer.vectorize(text = corpus.first())

        assertEquals(expected = base.dimension(), actual = vectorizer.dimension())
        assertEquals(expected = raw.indices.toList(), actual = weighted.indices.toList())
    }

    @Test
    internal fun `counts each document once however often a bucket repeats within it`() {
        val repeated = listOf("aa aa aa aa aa")

        val table = IdfVectorizer.fit(delegate = base, texts = repeated, minimumDocumentFrequency = 1).table()

        assertTrue(actual = table.frequencies.all { frequency -> frequency <= 1 })
    }

    /**
     * A refitted table changes every value in a vector while leaving every index alone, so nothing
     * downstream would notice the substitution. The fingerprint has to.
     */
    @Test
    internal fun `a table fitted on different data yields a different fingerprint`() {
        val first = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)
        val second = IdfVectorizer.fit(delegate = base, texts = corpus.drop(n = 1), minimumDocumentFrequency = 1)

        assertTrue(actual = first.fingerprint() != second.fingerprint())
        assertEquals(expected = "idf(hashing)", actual = first.fingerprint().kind)
    }

    @Test
    internal fun `an identical fit yields an identical fingerprint`() {
        val first = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)
        val second = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)

        assertEquals(expected = first.fingerprint(), actual = second.fingerprint())
    }

    @Test
    internal fun `differs from the plain hashing fingerprint it wraps`() {
        val vectorizer = IdfVectorizer.fit(delegate = base, texts = corpus, minimumDocumentFrequency = 1)

        assertTrue(actual = vectorizer.fingerprint() != base.fingerprint())
    }

    @Test
    internal fun `scales the floor with the corpus so two in a million is not kept`() {
        assertEquals(expected = 2, actual = DocumentFrequencyTable.floorFor(documentCount = 7_319))
        assertEquals(expected = 2, actual = DocumentFrequencyTable.floorFor(documentCount = 100_000))
        assertEquals(expected = 20, actual = DocumentFrequencyTable.floorFor(documentCount = 1_000_000))
        assertEquals(expected = 2, actual = DocumentFrequencyTable.floorFor(documentCount = 1))
    }

    @Test
    internal fun `reports a bucket outside the table as dropped`() {
        val table = DocumentFrequencyTable(
            documentCount = 10,
            frequencies = intArrayOf(5, 0),
            minimumDocumentFrequency = 2,
        )

        assertTrue(actual = table.isKept(bucket = 0))
        assertTrue(actual = !table.isKept(bucket = 1))
        assertTrue(actual = !table.isKept(bucket = 99))
        assertEquals(expected = 0.0, actual = table.idf(bucket = 99))
        assertEquals(expected = 1, actual = table.keptBucketCount())
    }

    @Test
    internal fun `rejects an empty fitting corpus or an invalid table`() {
        assertFailsWith<IllegalArgumentException> { IdfVectorizer.fit(delegate = base, texts = emptyList()) }
        assertFailsWith<IllegalArgumentException> {
            DocumentFrequencyTable(documentCount = 0, frequencies = intArrayOf(1), minimumDocumentFrequency = 1)
        }
        assertFailsWith<IllegalArgumentException> {
            DocumentFrequencyTable(documentCount = 1, frequencies = intArrayOf(), minimumDocumentFrequency = 1)
        }
        assertFailsWith<IllegalArgumentException> {
            DocumentFrequencyTable(documentCount = 1, frequencies = intArrayOf(1), minimumDocumentFrequency = 0)
        }
    }
}
