package io.skein.examples.localai

import io.skein.classify.application.ModelStore
import io.skein.classify.application.MultiLabelCrossValidator
import io.skein.classify.application.VectorizerCanaryException
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.MultiLabeledText
import io.skein.classify.domain.Schema
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.embedding.http.domain.EmbeddingProbes
import io.skein.classify.embedding.http.domain.EmbeddingServiceConfig
import io.skein.classify.embedding.http.infrastructure.HttpEmbeddingVectorizer
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier
import io.skein.classify.spi.Vectorizer
import io.skein.classify.spi.vectorizeAll
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.deleteIfExists
import kotlin.math.sqrt
import kotlin.random.Random
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * The behaviour the [localai][runLocalAiExample] example demonstrates, asserted rather than
 * printed — so the guarantees it teaches cannot quietly stop being true.
 *
 * Everything here runs against [StubEmbeddingServer] on a loopback port, with no model download,
 * no service to install and no network. That is the point of keeping the stub in `main`: a canary
 * only matters when a live service changes its weights underneath a trained model, and a stub is
 * the only way to stage that in a test suite.
 */
internal class LocalAiVerificationTest {

    private lateinit var stub: StubEmbeddingServer
    private lateinit var modelFile: Path

    @BeforeTest
    internal fun start() {
        stub = StubEmbeddingServer().start()
        modelFile = Files.createTempFile("localai-test", ".skein")
    }

    @AfterTest
    internal fun stop() {
        stub.close()
        modelFile.deleteIfExists()
    }

    private fun config(
        probes: List<String> = EmbeddingProbes.DEFAULT,
        tolerance: Double = VectorizerCanary.DEFAULT_TOLERANCE,
    ): EmbeddingServiceConfig {
        return EmbeddingServiceConfig(
            baseUrl = stub.baseUrl,
            model = "stub-embedding-model",
            modelRevision = "stub@1",
            dimension = stub.dimension,
            canaryProbes = probes,
            canaryTolerance = tolerance,
            batchSize = 32,
        )
    }

    private fun vectorizer(probes: List<String> = EmbeddingProbes.DEFAULT): HttpEmbeddingVectorizer {
        return HttpEmbeddingVectorizer(config = config(probes = probes))
    }

    /** A small deterministic multi-label corpus; the labels follow the keywords. */
    private fun corpus(): List<MultiLabeledText> {
        val random = Random(seed = 20260920)
        val rules = mapOf("crash" to "BUG", "login" to "AUTH", "slow" to "PERFORMANCE", "invoice" to "BILLING")
        val keywords = rules.keys.toList()
        return (0 until 40).flatMap { template ->
            val chosen = buildSet {
                add(keywords[random.nextInt(keywords.size)])
                if (random.nextInt(100) < 30) add(keywords[random.nextInt(keywords.size)])
            }
            val labels = chosen.map { keyword -> Label(value = rules.getValue(keyword)) }.toSet()
            (0 until 4).map { example ->
                MultiLabeledText(
                    featureText = "${chosen.joinToString(separator = " ")} ticket $template-$example",
                    labels = labels,
                    group = "template-$template",
                )
            }
        }
    }

    private val schema: Schema = Schema.define {
        text(name = "body")
        label(name = "tags")
    }

    private fun train(vectorizer: Vectorizer, rows: List<MultiLabeledText> = corpus()): MultiLabelLogisticClassifier {
        val vectors = vectorizer.vectorizeAll(texts = rows.map { row -> row.featureText })
        return LbfgsMultiLabelLearner(featureCount = vectorizer.dimension(), keepFraction = 1.0).fit(
            observations = rows.indices.map { index ->
                MultiLabeledFeatures(features = vectors[index], labels = rows[index].labels)
            },
        ) as MultiLabelLogisticClassifier
    }

    private fun save(vectorizer: Vectorizer) {
        ModelStore.saveMultiLabel(
            path = modelFile,
            schema = schema,
            model = train(vectorizer = vectorizer),
            vectorizer = vectorizer,
        )
    }

    @Test
    internal fun `trains, saves and reloads over a real socket`() {
        val subject = vectorizer()
        save(vectorizer = subject)

        val loaded = ModelStore.loadMultiLabel(path = modelFile, vectorizer = vectorizer())

        assertTrue(actual = Files.size(modelFile) > 0)
        assertEquals(expected = EmbeddingProbes.DEFAULT.size, actual = loaded.canary!!.probes.size)
        assertEquals(expected = stub.dimension, actual = loaded.canary!!.dimension())
    }

    /** The probes must cross the wire on load, or the check is decorative. */
    @Test
    internal fun `loading re-embeds the probes, and verifyCanary false does not`() {
        save(vectorizer = vectorizer())

        stub.resetCounters()
        ModelStore.loadMultiLabel(path = modelFile, vectorizer = vectorizer())
        val verified = stub.requests.get()

        stub.resetCounters()
        ModelStore.loadMultiLabel(path = modelFile, vectorizer = vectorizer(), verifyCanary = false)

        assertEquals(expected = EmbeddingProbes.DEFAULT.size, actual = verified)
        assertEquals(expected = 0, actual = stub.requests.get())
    }

    /**
     * The one the whole feature exists for. Nothing the client can see changes — same URL, model
     * name, declared revision and width — so the fingerprint still matches and only re-embedding
     * the probes can notice.
     */
    @Test
    internal fun `a model swapped on the server is refused at load`() {
        val subject = vectorizer()
        save(vectorizer = subject)

        stub.seed = 999L
        val afterSwap = vectorizer()

        assertEquals(
            expected = subject.fingerprint(),
            actual = afterSwap.fingerprint(),
            message = "the fingerprint changed, so this did not exercise the canary",
        )
        val failure = assertFailsWith<VectorizerCanaryException> {
            ModelStore.loadMultiLabel(path = modelFile, vectorizer = afterSwap)
        }
        assertTrue(actual = failure.drift > VectorizerCanary.DEFAULT_TOLERANCE)
    }

    /** Why drift is relative L2 and not cosine distance. */
    @Test
    internal fun `a service that starts normalising is caught, which cosine would miss`() {
        save(vectorizer = vectorizer())

        stub.scale = 1.5f
        val rescaled = vectorizer()
        val stored = ModelStore.loadMultiLabel(path = modelFile, vectorizer = rescaled, verifyCanary = false).canary!!
        val observed = stored.probes.map { probe -> rescaled.vectorize(text = probe).values }

        assertTrue(
            actual = cosineDistance(left = stored.references.first(), right = observed.first()) < 1e-6,
            message = "rescaling should leave direction untouched, which is what makes cosine blind to it",
        )
        assertFailsWith<VectorizerCanaryException> {
            ModelStore.loadMultiLabel(path = modelFile, vectorizer = vectorizer())
        }
    }

    /** A canary captured before a long run says nothing about a model swapped during it. */
    @Test
    internal fun `saving refuses a model that changed mid-training-run`() {
        val subject = vectorizer()
        subject.canary()
        val model = train(vectorizer = subject)

        stub.seed = 4242L

        assertFailsWith<VectorizerCanaryException> {
            ModelStore.saveMultiLabel(path = modelFile, schema = schema, model = model, vectorizer = subject)
        }
    }

    /** Cross-validation featurises every row once per fold; batching makes that one call. */
    @Test
    internal fun `cross-validation batches instead of asking row by row`() {
        val rows = corpus()

        stub.resetCounters()
        MultiLabelCrossValidator().crossValidate(
            corpus = rows,
            vectorizerFactory = { vectorizer(probes = emptyList()) },
            learnerFactory = { LbfgsMultiLabelLearner(featureCount = stub.dimension, keepFraction = 1.0) },
            folds = 5,
        )
        val batched = stub.requests.get()

        stub.resetCounters()
        MultiLabelCrossValidator().crossValidate(
            corpus = rows,
            vectorizerFactory = { Unbatched(inner = vectorizer(probes = emptyList())) },
            learnerFactory = { LbfgsMultiLabelLearner(featureCount = stub.dimension, keepFraction = 1.0) },
            folds = 5,
        )

        assertTrue(
            actual = stub.requests.get() > batched * 10,
            message = "batching saved almost nothing: $batched vs ${stub.requests.get()}",
        )
    }

    @Test
    internal fun `a multi-label file is refused by the single-label reader`() {
        save(vectorizer = vectorizer(probes = emptyList()))

        val failure = assertFailsWith<IllegalArgumentException> { ModelStore.load(path = modelFile) }

        assertTrue(actual = failure.message!!.contains(other = "loadMultiLabel"))
    }

    @Test
    internal fun `the calibrator reports a clean noise floor and a real drift`() {
        val calibrator = CanaryCalibrator()
        val subject = vectorizer()

        val floor = calibrator.noiseFloor(
            vectorizer = subject,
            probes = EmbeddingProbes.DEFAULT,
            reader = vectorizer(),
        )
        assertTrue(actual = !floor.trips, message = "an unchanged stub drifted: ${floor.drift}")
        assertTrue(actual = calibrator.verdict(reading = floor).contains(other = "unchanged"))

        val captured = subject.canary()!!
        stub.seed = 31337L
        val moved = calibrator.driftAgainst(canary = captured, vectorizer = vectorizer())

        assertTrue(actual = moved.trips)
        assertTrue(actual = calibrator.verdict(reading = moved).contains(other = "DRIFTED"))
    }

    /** Discovery has to survive the server simply not being there. */
    @Test
    internal fun `discovery reports nothing, and says why, when no server answers`() {
        val dead = "http://localhost:9/v1"

        assertTrue(actual = ModelDiscovery.advertised(baseUrl = dead).isEmpty())
        assertEquals(expected = null, actual = ModelDiscovery.firstEmbeddingModel(baseUrl = dead))
        assertTrue(actual = ModelDiscovery.diagnose(baseUrl = dead).contains(other = "No OpenAI-compatible server"))
    }

    /** The stub is deterministic, which every measurement above depends on. */
    @Test
    internal fun `the stub server is deterministic and its seed changes the model`() {
        val first = stub.embed(text = "alpha beta")
        val again = stub.embed(text = "alpha beta")
        assertTrue(actual = first.contentEquals(other = again))

        stub.seed = 2L
        assertTrue(actual = !first.contentEquals(other = stub.embed(text = "alpha beta")))
    }

    /** Strips the batch capability so the fallback path can be compared against it. */
    private class Unbatched(private val inner: Vectorizer) : Vectorizer {
        override fun vectorize(text: String): FeatureVector = inner.vectorize(text = text)
        override fun dimension(): Int = inner.dimension()
        override fun fingerprint(): VectorizerFingerprint = inner.fingerprint()
    }

    private fun cosineDistance(left: FloatArray, right: FloatArray): Double {
        var dot = 0.0
        var leftNorm = 0.0
        var rightNorm = 0.0
        for (index in left.indices) {
            dot += left[index].toDouble() * right[index].toDouble()
            leftNorm += left[index].toDouble() * left[index].toDouble()
            rightNorm += right[index].toDouble() * right[index].toDouble()
        }
        return 1.0 - dot / (sqrt(x = leftNorm) * sqrt(x = rightNorm))
    }
}
