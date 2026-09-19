package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelThresholds
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.Schema
import io.skein.classify.domain.TermWeightingEnum
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

internal class MultiLabelModelStoreTest {

    private val featureCount = 1 shl 12
    private val path: Path = Files.createTempFile("multi-label-model", ".skein")

    @AfterTest
    internal fun cleanUp() {
        path.deleteIfExists()
    }

    private val schema: Schema = Schema.define {
        text(name = "title")
        text(name = "body")
        label(name = "tags")
    }

    private val hashingConfig = HashingConfig(key0 = 7L, key1 = 11L, numFeatures = featureCount)

    private fun vectorizer(config: HashingConfig = hashingConfig): Vectorizer {
        return HashingVectorizer(config = config)
    }

    private fun features(vararg indices: Int): FeatureVector {
        return FeatureVector(indices = indices.sortedArray(), values = FloatArray(size = indices.size) { 1.0f })
    }

    private fun corpus(): List<MultiLabeledFeatures> {
        val alpha = Label(value = "ALPHA")
        val beta = Label(value = "BETA")
        return listOf(
            MultiLabeledFeatures(features = features(1, 20), labels = setOf(alpha)),
            MultiLabeledFeatures(features = features(1, 21), labels = setOf(alpha)),
            MultiLabeledFeatures(features = features(2, 20), labels = setOf(beta)),
            MultiLabeledFeatures(features = features(2, 22), labels = setOf(beta)),
            MultiLabeledFeatures(features = features(1, 2), labels = setOf(alpha, beta)),
            MultiLabeledFeatures(features = features(3, 23), labels = emptySet()),
        )
    }

    private fun trainedModel(): MultiLabelLogisticClassifier {
        return LbfgsMultiLabelLearner(
            featureCount = featureCount,
            gradientTolerance = 1e-8,
            keepFraction = 1.0,
        ).fit(observations = corpus()) as MultiLabelLogisticClassifier
    }

    private val probes = listOf(features(1, 20), features(2, 22), features(1, 2), features(3, 23), features(99))

    /**
     * The single most valuable persistence test: train, save, load, and confirm the restored model
     * scores identically. Everything else about the format is detail.
     *
     * The tolerance is `float16`'s, not zero, because the weights are stored in half precision by
     * design. That is the measured trade the encoding makes — see [io.skein.classify.domain.WeightEncodingEnum].
     */
    @Test
    internal fun `a saved model scores a record the same after loading`() {
        val original = trainedModel()
        ModelStore.saveMultiLabel(
            path = path,
            schema = schema,
            model = original,
            vectorizer = vectorizer(),
            hashingConfig = hashingConfig,
        )

        val restored = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).classifier

        probes.forEach { probe ->
            val before = original.logits(features = probe)
            val after = restored.logits(features = probe)
            assertEquals(expected = before.keys, actual = after.keys)
            before.forEach { (label, logit) ->
                assertEquals(
                    expected = logit,
                    actual = after.getValue(key = label),
                    absoluteTolerance = 1e-2,
                    message = "$label drifted on $probe",
                )
            }
        }
    }

    @Test
    internal fun `a saved model predicts the same label sets after loading`() {
        val original = trainedModel()
        ModelStore.saveMultiLabel(path = path, schema = schema, model = original, vectorizer = vectorizer())

        val restored = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).classifier

        probes.forEach { probe ->
            assertEquals(
                expected = original.predict(features = probe, threshold = 0.5).labels(),
                actual = restored.predict(features = probe, threshold = 0.5).labels(),
                message = "label set changed for $probe",
            )
        }
    }

    /**
     * The safety mechanism the whole fingerprint exists for. A model scored with a different
     * featurisation returns confident, wrong labels and throws nothing on its own.
     */
    @Test
    internal fun `refuses to load a model with a different hashing key`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        val wrongKey = HashingConfig(key0 = 999L, key1 = 11L, numFeatures = featureCount)

        val failure = assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer(config = wrongKey))
        }
        assertTrue(actual = failure.message!!.contains(other = "confident but wrong"))
    }

    @Test
    internal fun `refuses to load a model with a different feature width or n-gram range`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(
                path = path,
                vectorizer = vectorizer(config = hashingConfig.copy(numFeatures = featureCount * 2)),
            )
        }
        assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(
                path = path,
                vectorizer = vectorizer(config = hashingConfig.copy(charNgramMax = 6)),
            )
        }
    }

    @Test
    internal fun `refuses to load a model whose term weighting changed`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(
                path = path,
                vectorizer = vectorizer(
                    config = hashingConfig.copy(termWeighting = TermWeightingEnum.SUBLINEAR),
                ),
            )
        }
    }

    @Test
    internal fun `accepts a vectorizer that is a different instance of the same configuration`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        val loaded = ModelStore.loadMultiLabel(
            path = path,
            vectorizer = HashingVectorizer(config = HashingConfig(key0 = 7L, key1 = 11L, numFeatures = featureCount)),
        )

        assertEquals(expected = "hashing", actual = loaded.fingerprint.kind)
    }

    @Test
    internal fun `restores the schema and the hashing configuration`() {
        ModelStore.saveMultiLabel(
            path = path,
            schema = schema,
            model = trainedModel(),
            vectorizer = vectorizer(),
            hashingConfig = hashingConfig,
        )

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer())

        assertEquals(expected = listOf("title", "body", "tags"), actual = loaded.schema.fields.map { it.name })
        assertEquals(expected = hashingConfig, actual = loaded.hashingConfig)
    }

    @Test
    internal fun `omits the hashing configuration when the model was trained without one`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        assertNull(actual = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).hashingConfig)
    }

    @Test
    internal fun `round-trips per-label thresholds`() {
        val thresholds = LabelThresholds(
            byLabel = mapOf(Label(value = "ALPHA") to 0.8, Label(value = "BETA") to 0.25),
            fallback = 0.4,
        )
        ModelStore.saveMultiLabel(
            path = path,
            schema = schema,
            model = trainedModel(),
            vectorizer = vectorizer(),
            thresholds = thresholds,
        )

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).thresholds

        assertEquals(expected = 0.8, actual = loaded.of(label = Label(value = "ALPHA")))
        assertEquals(expected = 0.25, actual = loaded.of(label = Label(value = "BETA")))
        assertEquals(expected = 0.4, actual = loaded.of(label = Label(value = "NEVER_FITTED")))
    }

    @Test
    internal fun `defaults to a uniform threshold when none is given`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        val loaded = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).thresholds

        assertEquals(expected = 0.5, actual = loaded.of(label = Label(value = "ALPHA")))
        assertTrue(actual = loaded.tunedLabels().isEmpty())
    }

    /**
     * A corpus built through the vectorizer, so several hundred features per row are active and the
     * fitted matrix is genuinely dense. The toy corpus above cannot show this: with only a handful
     * of features ever seen, almost every weight is exactly zero already and there is nothing for
     * pruning to remove.
     */
    private fun textCorpus(): List<MultiLabeledFeatures> {
        val vectorizer = vectorizer()
        val rows = listOf(
            "slow roasted tomato risotto with parmesan and basil" to setOf("ITALIAN"),
            "pancetta carbonara pasta with black pepper" to setOf("ITALIAN"),
            "chipotle bean burrito with tortilla and jalapeno" to setOf("MEXICAN"),
            "grilled quesadilla with tortilla and cheese" to setOf("MEXICAN"),
            "basil tomato tortilla flatbread with parmesan" to setOf("ITALIAN", "MEXICAN"),
            "boiled rice with water and salt" to emptySet(),
        )
        return rows.map { (text, labels) ->
            MultiLabeledFeatures(
                features = vectorizer.vectorize(text = text),
                labels = labels.map { value -> Label(value = value) }.toSet(),
            )
        }
    }

    private fun textModel(keepFraction: Double): MultiLabelLogisticClassifier {
        return LbfgsMultiLabelLearner(
            featureCount = featureCount,
            gradientTolerance = 1e-8,
            keepFraction = keepFraction,
        ).fit(observations = textCorpus()) as MultiLabelLogisticClassifier
    }

    @Test
    internal fun `pruning drops weights and shrinks the file`() {
        val dense = textModel(keepFraction = 1.0)
        val pruned = textModel(keepFraction = 0.05)

        assertTrue(
            actual = pruned.weights().nonZeroCount() < dense.weights().nonZeroCount(),
            message = "${pruned.weights().nonZeroCount()} was not below ${dense.weights().nonZeroCount()}",
        )

        ModelStore.saveMultiLabel(path = path, schema = schema, model = dense, vectorizer = vectorizer())
        val denseSize = Files.size(path)
        ModelStore.saveMultiLabel(path = path, schema = schema, model = pruned, vectorizer = vectorizer())
        val prunedSize = Files.size(path)

        assertTrue(actual = prunedSize < denseSize, message = "$prunedSize was not below $denseSize")
    }

    /**
     * The compact encoding is only worth having if it survives a round trip, so pin the drift
     * rather than trusting it. `float16` carries about three decimal digits, which is far more than
     * a pruned weight matrix needs.
     */
    @Test
    internal fun `half-precision storage keeps scores within its documented drift`() {
        val model = textModel(keepFraction = 1.0)
        val probe = vectorizer().vectorize(text = "tomato risotto with parmesan")
        ModelStore.saveMultiLabel(path = path, schema = schema, model = model, vectorizer = vectorizer())

        val restored = ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer()).classifier

        val before = model.scoreAll(features = probe)
        val after = restored.scoreAll(features = probe)
        before.forEach { (label, probability) ->
            assertEquals(
                expected = probability,
                actual = after.getValue(key = label),
                absoluteTolerance = 1e-3,
                message = "$label drifted beyond half-precision tolerance",
            )
        }
    }

    /** A single-label reader must reject a multi-label file by name rather than misread it. */
    @Test
    internal fun `the single-label loader refuses a multi-label file and says where to go`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())

        val failure = assertFailsWith<IllegalArgumentException> { ModelStore.load(path = path) }

        assertTrue(
            actual = failure.message!!.contains(other = "loadMultiLabel"),
            message = failure.message!!,
        )
    }

    @Test
    internal fun `the multi-label loader refuses a single-label file`() {
        ModelStore.save(
            path = path,
            schema = schema,
            classifier = ClassifierKindEnum.LOGISTIC_REGRESSION,
            hashingConfig = hashingConfig,
            observations = emptyList(),
        )

        assertFailsWith<IllegalArgumentException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer())
        }
    }

    @Test
    internal fun `reports a corrupt file as one exception type`() {
        Files.write(path, byteArrayOf(0x53, 0x4B, 0x45, 0x49, 0x03, 0x00, 0x01))

        assertFailsWith<IllegalArgumentException> {
            ModelStore.loadMultiLabel(path = path, vectorizer = vectorizer())
        }
    }

    /**
     * The one piece of the featurisation a caller cannot reconstruct from its own configuration: a
     * fitted document-frequency table exists only as an artifact of the corpus it was fitted on. It
     * therefore travels in the model file, and the factory overload is how it gets back out.
     */
    @Test
    internal fun `round-trips a fitted document-frequency table and scores identically`() {
        val texts = listOf(
            "slow roasted tomato risotto with parmesan",
            "pancetta carbonara pasta with black pepper",
            "chipotle bean burrito with tortilla",
            "grilled quesadilla with tortilla and cheese",
        )
        val idf = IdfVectorizer.fit(delegate = vectorizer(), texts = texts, minimumDocumentFrequency = 1)
        val model = LbfgsMultiLabelLearner(
            featureCount = featureCount,
            gradientTolerance = 1e-8,
            keepFraction = 1.0,
        ).fit(
            observations = texts.mapIndexed { index, text ->
                MultiLabeledFeatures(
                    features = idf.vectorize(text = text),
                    labels = setOf(Label(value = if (index < 2) "ITALIAN" else "MEXICAN")),
                )
            },
        ) as MultiLabelLogisticClassifier

        ModelStore.saveMultiLabel(path = path, schema = schema, model = model, vectorizer = idf)
        val loaded = ModelStore.loadMultiLabel(
            path = path,
            vectorizerFactory = { table ->
                IdfVectorizer(delegate = vectorizer(), table = requireNotNull(value = table))
            },
        )

        val probe = idf.vectorize(text = "tomato risotto with parmesan")
        model.logits(features = probe).forEach { (label, logit) ->
            assertEquals(
                expected = logit,
                actual = loaded.classifier.logits(features = probe).getValue(key = label),
                absoluteTolerance = 1e-2,
            )
        }
    }

    /** A factory that ignores the stored table builds the wrong vectorizer, and must be caught. */
    @Test
    internal fun `refuses a factory that discards the stored document-frequency table`() {
        val texts = listOf("alpha beta gamma", "beta gamma delta", "gamma delta epsilon")
        val idf = IdfVectorizer.fit(delegate = vectorizer(), texts = texts, minimumDocumentFrequency = 1)
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = idf)

        assertFailsWith<VectorizerMismatchException> {
            ModelStore.loadMultiLabel(path = path, vectorizerFactory = { vectorizer() })
        }
    }

    @Test
    internal fun `hands the factory a null table when the model was trained without one`() {
        ModelStore.saveMultiLabel(path = path, schema = schema, model = trainedModel(), vectorizer = vectorizer())
        var seen: Any? = "not called"

        ModelStore.loadMultiLabel(
            path = path,
            vectorizerFactory = { table ->
                seen = table
                vectorizer()
            },
        )

        assertNull(actual = seen)
    }

    @Test
    internal fun `a fingerprint identifies the configuration rather than the instance`() {
        val first = vectorizer().fingerprint()
        val second = vectorizer().fingerprint()

        assertEquals(expected = first, actual = second)
        assertEquals(
            expected = VectorizerFingerprint(
                kind = "hashing",
                dimension = featureCount,
                configDigest = first.configDigest,
            ),
            actual = first,
        )
    }
}
