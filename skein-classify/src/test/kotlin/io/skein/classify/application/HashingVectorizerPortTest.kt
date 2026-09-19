package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.TermWeightingEnum
import io.skein.text.spi.TextNormalizer
import kotlin.math.ln
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/** The [io.skein.classify.spi.Vectorizer] surface and the term weighting, separate from hashing itself. */
internal class HashingVectorizerPortTest {

    private val config = HashingConfig(key0 = 13L, key1 = 17L, numFeatures = 1 shl 12)

    private fun vectorizer(config: HashingConfig = this.config): HashingVectorizer {
        return HashingVectorizer(config = config)
    }

    @Test
    internal fun `reports the configured feature space as its dimension`() {
        assertEquals(expected = 1 shl 12, actual = vectorizer().dimension())
    }

    @Test
    internal fun `raw counts report how often an n-gram occurred`() {
        val vector = vectorizer().vectorize(text = "ab ab ab")

        assertTrue(
            actual = vector.values.any { value -> value >= 3.0f },
            message = "expected a repeated n-gram to reach a count of three",
        )
    }

    @Test
    internal fun `sublinear weighting damps a repeated n-gram`() {
        val raw = vectorizer().vectorize(text = "ab ab ab")
        val sublinear = vectorizer(config = config.copy(termWeighting = TermWeightingEnum.SUBLINEAR))
            .vectorize(text = "ab ab ab")

        assertEquals(expected = raw.indices.toList(), actual = sublinear.indices.toList())
        raw.indices.indices.forEach { position ->
            assertEquals(
                expected = (1.0 + ln(x = raw.values[position].toDouble())).toFloat(),
                actual = sublinear.values[position],
                absoluteTolerance = 1e-6f,
            )
        }
    }

    @Test
    internal fun `sublinear weighting leaves a single occurrence at one`() {
        val text = "the quick brown fox jumps over a lazy dog"
        val raw = vectorizer().vectorize(text = text)
        val sublinear = vectorizer(config = config.copy(termWeighting = TermWeightingEnum.SUBLINEAR))
            .vectorize(text = text)

        val singles = raw.values.indices.filter { position -> raw.values[position] == 1.0f }
        assertTrue(actual = singles.isNotEmpty(), message = "expected some n-gram to occur exactly once")
        singles.forEach { position ->
            assertEquals(expected = 1.0f, actual = sublinear.values[position])
        }
    }

    @Test
    internal fun `sublinear weighting compresses the spread between rare and repeated n-grams`() {
        val text = "ab ab ab ab ab ab ab ab cd"
        val raw = vectorizer().vectorize(text = text)
        val sublinear = vectorizer(config = config.copy(termWeighting = TermWeightingEnum.SUBLINEAR))
            .vectorize(text = text)

        assertTrue(
            actual = sublinear.values.max() / sublinear.values.min() < raw.values.max() / raw.values.min(),
            message = "sublinear weighting should narrow the ratio between the most and least frequent",
        )
    }

    @Test
    internal fun `the fingerprint identifies the configuration, not the instance`() {
        assertEquals(expected = vectorizer().fingerprint(), actual = vectorizer().fingerprint())
    }

    @Test
    internal fun `every setting that changes a vector changes the fingerprint`() {
        val baseline = vectorizer().fingerprint()
        val variants = listOf(
            config.copy(key0 = 99L),
            config.copy(key1 = 99L),
            config.copy(numFeatures = 1 shl 13),
            config.copy(charNgramMin = 2),
            config.copy(charNgramMax = 6),
            config.copy(wordNgramMin = 2, wordNgramMax = 3),
            config.copy(termWeighting = TermWeightingEnum.SUBLINEAR),
        )

        variants.forEach { variant ->
            assertTrue(
                actual = vectorizer(config = variant).fingerprint() != baseline,
                message = "$variant shared a fingerprint with the baseline",
            )
        }
    }

    /**
     * A normalizer that folds text differently produces different features from the same input, so
     * it has to be part of the identity even though it is not part of [HashingConfig].
     */
    @Test
    internal fun `a different normalizer changes the fingerprint`() {
        val shouting = object : TextNormalizer {
            override fun normalize(raw: String): String {
                return raw.uppercase()
            }
        }

        assertTrue(
            actual = HashingVectorizer(config = config, normalizer = shouting).fingerprint() !=
                vectorizer().fingerprint(),
        )
    }

    @Test
    internal fun `the fingerprint reports the feature width it was built from`() {
        assertEquals(expected = 1 shl 12, actual = vectorizer().fingerprint().dimension)
        assertEquals(expected = "hashing", actual = vectorizer().fingerprint().kind)
    }
}
