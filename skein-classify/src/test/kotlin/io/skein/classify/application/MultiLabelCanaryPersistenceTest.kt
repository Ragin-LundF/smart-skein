package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.Schema
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier
import io.skein.classify.spi.Vectorizer
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists
import kotlin.test.AfterTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * The canary end to end through the `.skein` format: captured at save, re-checked at load.
 *
 * A [VectorizerFingerprint] cannot catch a vectorizer whose *configuration* is unchanged but whose
 * *output* is not — which is the permanent condition of an external embedding service. These tests
 * use a stub vectorizer whose output can be moved on demand, so the whole mechanism is exercised
 * with no HTTP anywhere.
 */
internal class MultiLabelCanaryPersistenceTest {

    private val featureCount = 8
    private val path: Path = Files.createTempFile("canary-model", ".skein")

    @AfterTest
    internal fun cleanUp() {
        path.deleteIfExists()
    }

    private val schema: Schema = Schema.define {
        text(name = "title")
        label(name = "tags")
    }

    /**
     * A vectorizer whose declared identity is fixed but whose vectors can be moved, which is
     * exactly the situation the canary exists for.
     *
     * @param scale multiplies every component — a service that started L2-normalising.
     * @param shift added to the first component — a service serving different weights.
     */
    private class StubVectorizer(
        private val width: Int,
        private val probes: List<String> = listOf("alpha", "beta"),
        private val scale: Float = 1.0f,
        private val shift: Float = 0.0f,
        private val digest: String = "fixed-digest",
    ) : Vectorizer {

        var vectorizeCalls: Int = 0
            private set

        override fun vectorize(text: String): FeatureVector {
            vectorizeCalls += 1
            val values = FloatArray(size = width) { index ->
                ((text.length + index + 1).toFloat() * scale) + if (index == 0) shift else 0.0f
            }
            return FeatureVector(indices = IntArray(size = width) { it }, values = values)
        }

        override fun dimension(): Int = width

        override fun fingerprint(): VectorizerFingerprint {
            return VectorizerFingerprint(kind = "stub", dimension = width, configDigest = digest)
        }

        override fun canary(): VectorizerCanary? {
            if (probes.isEmpty()) {
                return null
            }
            return VectorizerCanary(
                probes = probes,
                references = probes.map { probe -> vectorize(text = probe).values },
            )
        }
    }

    private fun trainedModel(): MultiLabelLogisticClassifier {
        val alpha = Label(value = "ALPHA")
        val beta = Label(value = "BETA")
        val corpus = listOf(
            MultiLabeledFeatures(features = features(0, 1), labels = setOf(alpha)),
            MultiLabeledFeatures(features = features(2, 3), labels = setOf(beta)),
            MultiLabeledFeatures(features = features(0, 3), labels = setOf(alpha, beta)),
            MultiLabeledFeatures(features = features(4, 5), labels = emptySet()),
        )
        return LbfgsMultiLabelLearner(featureCount = featureCount, keepFraction = 1.0)
            .fit(observations = corpus) as MultiLabelLogisticClassifier
    }

    private fun features(vararg indices: Int): FeatureVector {
        return FeatureVector(indices = indices.sortedArray(), values = FloatArray(size = indices.size) { 1.0f })
    }

    private fun save(vectorizer: Vectorizer) {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer)
    }

    /** A model saved by a vectorizer offering no canary must behave exactly as it always did. */
    @Test
    internal fun `a model without a canary loads without one, and embeds nothing on load`() {
        save(vectorizer = StubVectorizer(width = featureCount, probes = emptyList()))
        val loader = StubVectorizer(width = featureCount, probes = emptyList())

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = loader)

        assertNull(actual = loaded.canary)
        assertEquals(expected = 0, actual = loader.vectorizeCalls)
    }

    @Test
    internal fun `a canary round-trips and verifies against the same vectorizer`() {
        save(vectorizer = StubVectorizer(width = featureCount))

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = StubVectorizer(width = featureCount))

        assertEquals(expected = listOf("alpha", "beta"), actual = loaded.canary!!.probes)
        assertEquals(expected = featureCount, actual = loaded.canary!!.dimension())
        assertTrue(actual = loaded.canary!!.matches(observed = loaded.canary!!.references))
    }

    @Test
    internal fun `drift below the tolerance still loads`() {
        save(vectorizer = StubVectorizer(width = featureCount))

        val loaded = ModelStore.loadMultiLabel(
            path = path,
            vectorizer = StubVectorizer(width = featureCount, shift = 0.000001f),
        )

        assertEquals(expected = 2, actual = loaded.canary!!.probes.size)
    }

    @Test
    internal fun `a model changed underneath the fingerprint is refused at load`() {
        save(vectorizer = StubVectorizer(width = featureCount))

        val failure = assertFailsWith<VectorizerCanaryException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = StubVectorizer(width = featureCount, shift = 5.0f))
        }

        // Both probes shift by the same absolute amount, so the one with the smaller reference
        // norm shows the larger *relative* drift. That is the metric behaving as intended.
        assertEquals(expected = "beta", actual = failure.probe)
        assertTrue(actual = failure.drift > VectorizerCanary.DEFAULT_TOLERANCE)
        assertTrue(actual = failure.message!!.contains(other = "fingerprint has not"))
    }

    /** Relative L2 rather than cosine: a rescaled vector is a broken model for a linear classifier. */
    @Test
    internal fun `a service that starts normalising its output is caught`() {
        save(vectorizer = StubVectorizer(width = featureCount))

        assertFailsWith<VectorizerCanaryException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = StubVectorizer(width = featureCount, scale = 1.5f))
        }
    }

    @Test
    internal fun `verifyCanary false skips the check and the round trips it costs`() {
        save(vectorizer = StubVectorizer(width = featureCount))
        val loader = StubVectorizer(width = featureCount, shift = 5.0f)

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = loader, verifyCanary = false)

        assertEquals(expected = 0, actual = loader.vectorizeCalls)
        assertEquals(expected = 2, actual = loaded.canary!!.probes.size)
    }

    /**
     * Ordering matters: the fingerprint is free and the canary costs a network round trip per
     * probe, so an outright wrong vectorizer must fail before paying for any of them.
     */
    @Test
    internal fun `a fingerprint mismatch fails before any probe is embedded`() {
        save(vectorizer = StubVectorizer(width = featureCount))
        val loader = StubVectorizer(width = featureCount, digest = "different")

        assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = loader)
        }
        assertEquals(expected = 0, actual = loader.vectorizeCalls)
    }

    /**
     * A canary captured before training proves nothing about a model swapped *during* it: half the
     * corpus would be embedded by one model and half by another, and the stale reference would
     * still match at load. Re-checking at save is what closes that.
     */
    @Test
    internal fun `saving refuses a canary the vectorizer no longer reproduces`() {
        val drifting = object : Vectorizer {
            private var calls = 0

            override fun vectorize(text: String): FeatureVector {
                calls += 1
                val offset = if (calls > 2) 9.0f else 0.0f
                return FeatureVector(
                    indices = IntArray(size = featureCount) { it },
                    values = FloatArray(size = featureCount) { index -> (index + 1).toFloat() + offset },
                )
            }

            override fun dimension(): Int = featureCount

            override fun fingerprint(): VectorizerFingerprint {
                return VectorizerFingerprint(kind = "stub", dimension = featureCount, configDigest = "d")
            }

            override fun canary(): VectorizerCanary {
                return VectorizerCanary(
                    probes = listOf("alpha", "beta"),
                    references = listOf(vectorize(text = "alpha").values, vectorize(text = "beta").values),
                )
            }
        }

        assertFailsWith<VectorizerCanaryException> { save(vectorizer = drifting) }
    }
}
