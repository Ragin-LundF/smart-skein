@file:OptIn(ExperimentalSerializationApi::class)

package io.skein.classify.application

import io.skein.classify.domain.Calibration
import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.DocumentFrequencyTable
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.FieldSpec
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.LabelThresholds
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SchemaBuilder
import io.skein.classify.domain.SensitivityEnum
import io.skein.classify.domain.TermWeightingEnum
import io.skein.classify.domain.TextField
import io.skein.classify.domain.VectorizerCanary
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.domain.WeightEncodingEnum
import io.skein.classify.infrastructure.MultiLabelLogisticClassifier
import io.skein.classify.infrastructure.MultiLabelWeights
import io.skein.classify.infrastructure.WeightCodec
import io.skein.classify.spi.Vectorizer
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.protobuf.ProtoBuf
import java.io.EOFException
import java.nio.file.Path
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import java.util.zip.ZipException
import kotlin.io.path.inputStream
import kotlin.io.path.outputStream

// Layout: 4-byte magic 'SKEI' + 1-byte version, then GZIP(ProtoBuf payload).
//
// The version byte selects the PAYLOAD SHAPE, not merely a revision:
//   v1, v2 -> SkeinModelDto           - a corpus of observations, replayed through a classifier on load
//   v3     -> MultiLabelModelDto      - an already-fitted weight matrix
// A batch-fitted multi-label model has no incremental history to replay, so the observation shape
// cannot express it and a separate payload is the honest representation rather than a widening of
// the old one. v3 is a new shape, not a superset: an older reader rejects it at the header instead
// of misreading it.
//
// Within a shape, new fields must only ever be APPENDED to the DTO, since ProtoBuf field numbers
// here are positional; a v1 file decodes under v2 with the default applied.
// Privacy: the hashing key (keyed-PRF secret) and already-irreversible feature vectors are stored — treat as a secret.
private val MAGIC = byteArrayOf(0x53, 0x4B, 0x45, 0x49) // 'SKEI'
private const val VERSION: Byte = 0x02
private const val VERSION_V1: Byte = 0x01
private const val VERSION_MULTI_LABEL: Byte = 0x03
private const val HEADER_SIZE = 5

/** Byte offset of the version within the header, immediately after the four-byte magic. */
private const val VERSION_OFFSET = 4
private const val UNCALIBRATED_TEMPERATURE = 1.0

/** Acceptance threshold stored with a multi-label model when the caller names none. */
private const val DEFAULT_THRESHOLD = 0.5

private const val FIELD_TEXT = 0
private const val FIELD_CATEGORICAL = 1
private const val FIELD_NUMERIC = 2
private const val FIELD_IDENTIFIER = 3
private const val FIELD_LABEL = 4

@Serializable
private data class SkeinModelDto(
    val classifier: Int,
    val key0: Long,
    val key1: Long,
    val numFeatures: Int,
    val charNgramMin: Int,
    val charNgramMax: Int,
    val wordNgramMin: Int,
    val wordNgramMax: Int,
    val fields: List<FieldDto>,
    val observations: List<ObservationDto>,
    // Appended in v2; a v1 file omits it and decodes to this default.
    val calibrationTemperature: Double = UNCALIBRATED_TEMPERATURE,
    // Also appended in v2. Without these, a tuned model is replayed through a default-constructed
    // classifier on load and comes back as a different model.
    val smoothingAlpha: Double = ClassifierHyperparameters.DEFAULT_SMOOTHING_ALPHA,
    val initialLearningRate: Double = ClassifierHyperparameters.DEFAULT_LEARNING_RATE,
    val decayRate: Double = ClassifierHyperparameters.DEFAULT_DECAY_RATE,
    val l2Regularization: Double = ClassifierHyperparameters.DEFAULT_L2_REGULARIZATION,
)

@Serializable
private data class FieldDto(val type: Int, val name: String, val sensitivity: Int)

@Serializable
private class ObservationDto(val label: String, val indices: IntArray, val values: FloatArray)

/**
 * The v3 payload: a fitted one-vs-rest weight matrix.
 *
 * [labelIndexDeltas] and [weights] are the two sections that dominate the file, and both are stored
 * in their compact form — see [WeightCodec]. [hasHashingConfig] distinguishes a model trained with
 * this library's feature hashing from one trained with an external vectorizer, for which the
 * hashing fields carry nothing meaningful.
 */
@Serializable
private class MultiLabelModelDto(
    val fields: List<FieldDto>,
    val vectorizerKind: String,
    val vectorizerDimension: Int,
    val vectorizerDigest: String,
    val labels: List<String>,
    val intercepts: DoubleArray,
    val featureCount: Int,
    val featureOffsets: IntArray,
    val labelIndexDeltas: IntArray,
    val weightEncoding: Int,
    val weights: ByteArray,
    val l2Regularization: Double,
    val thresholdFallback: Double,
    val tunedThresholdLabels: List<String>,
    val tunedThresholds: DoubleArray,
    val hasHashingConfig: Boolean,
    val key0: Long,
    val key1: Long,
    val numFeatures: Int,
    val charNgramMin: Int,
    val charNgramMax: Int,
    val wordNgramMin: Int,
    val wordNgramMax: Int,
    val termWeighting: Int,
    // Appended: present only when the model was trained through an IdfVectorizer, whose fitted
    // table changes every feature value and therefore has to travel with the model.
    val hasDocumentFrequencies: Boolean = false,
    val documentCount: Int = 0,
    val minimumDocumentFrequency: Int = 1,
    val documentFrequencies: IntArray = IntArray(size = 0),
    // Appended: present only when the vectorizer offered a canary at save time. Held flat and
    // row-major, stride vectorizerDimension, for the same reason the weights are -- packed
    // ProtoBuf floats rather than a length prefix per row.
    val hasCanary: Boolean = false,
    val canaryProbes: List<String> = emptyList(),
    val canaryVectors: FloatArray = FloatArray(size = 0),
    val canaryTolerance: Double = VectorizerCanary.DEFAULT_TOLERANCE,
)

/** The four DTO fields a canary occupies, so building them adds no branches to a save. */
private class CanaryFields(
    val present: Boolean,
    val probes: List<String>,
    val vectors: FloatArray,
    val tolerance: Double,
)

object ModelStore {

    /**
     * Writes a model with no calibration and default hyperparameters. Retained as its own overload
     * so callers compiled against the five-argument form keep linking.
     */
    fun save(
        path: Path,
        schema: Schema,
        classifier: ClassifierKindEnum,
        hashingConfig: HashingConfig,
        observations: List<LabeledFeatures>,
    ) {
        save(
            path = path,
            schema = schema,
            classifier = classifier,
            hashingConfig = hashingConfig,
            observations = observations,
            calibration = Calibration.NONE,
        )
    }

    /**
     * Writes a model.
     *
     * [classifier] names the *kind* of model, so this cannot discover how the classifier was tuned —
     * pass [hyperparameters] to preserve it. A tuned model saved without them is replayed through a
     * default-constructed classifier on load and comes back behaving differently:
     *
     * ```kotlin
     * ModelStore.save(
     *     path = path,
     *     schema = engine.schema,
     *     classifier = ClassifierKindEnum.LOGISTIC_REGRESSION,
     *     hashingConfig = hashingConfig,
     *     observations = engine.featureStore.all(),
     *     calibration = engine.calibration,
     *     hyperparameters = engine.classifier.hyperparameters(),
     * )
     * ```
     */
    fun save(
        path: Path,
        schema: Schema,
        classifier: ClassifierKindEnum,
        hashingConfig: HashingConfig,
        observations: List<LabeledFeatures>,
        calibration: Calibration,
        hyperparameters: ClassifierHyperparameters = ClassifierHyperparameters.DEFAULTS,
    ) {
        val dto = SkeinModelDto(
            classifier = classifier.ordinal,
            key0 = hashingConfig.key0,
            key1 = hashingConfig.key1,
            numFeatures = hashingConfig.numFeatures,
            charNgramMin = hashingConfig.charNgramMin,
            charNgramMax = hashingConfig.charNgramMax,
            wordNgramMin = hashingConfig.wordNgramMin,
            wordNgramMax = hashingConfig.wordNgramMax,
            fields = schema.fields.map { field -> fieldDto(field = field) },
            observations = observations.map { obs ->
                ObservationDto(
                    label = obs.label.value,
                    indices = obs.features.indices,
                    values = obs.features.values,
                )
            },
            calibrationTemperature = calibration.temperature,
            smoothingAlpha = hyperparameters.smoothingAlpha,
            initialLearningRate = hyperparameters.initialLearningRate,
            decayRate = hyperparameters.decayRate,
            l2Regularization = hyperparameters.l2Regularization,
        )
        val encoded = ProtoBuf.encodeToByteArray(serializer = SkeinModelDto.serializer(), value = dto)
        path.outputStream().use { file ->
            file.write(MAGIC)
            file.write(VERSION.toInt())
            GZIPOutputStream(file).use { gzip -> gzip.write(encoded) }
        }
    }

    /**
     * Verifies the header and inflates the payload, reporting any damage as one exception type.
     *
     * [accepted] is the set of payload shapes the caller can decode, so a multi-label file handed
     * to [load] is refused by name rather than decoded as a corrupt observation corpus.
     */
    private fun readPayload(path: Path, accepted: List<Byte>): ByteArray {
        return path.inputStream().use { file ->
            val header = file.readNBytes(HEADER_SIZE)
            require(
                value = header.size == HEADER_SIZE && MAGIC.indices.all { i -> header[i] == MAGIC[i] },
            ) { "malformed .skein file (expected the SKEI binary format)" }
            val version = header[VERSION_OFFSET]
            require(value = version in accepted) {
                "this .skein file holds a version $version payload; " +
                    if (version == VERSION_MULTI_LABEL) {
                        "it is a multi-label model, so read it with ModelStore.loadMultiLabel"
                    } else {
                        "expected one of $accepted"
                    }
            }
            // A valid header followed by a truncated or corrupt payload would otherwise surface as a
            // raw EOFException/ZipException; keep load's contract to one exception type.
            try {
                GZIPInputStream(file).readBytes()
            } catch (cause: EOFException) {
                throw IllegalArgumentException("truncated .skein file", cause)
            } catch (cause: ZipException) {
                throw IllegalArgumentException("corrupt .skein file", cause)
            }
        }
    }

    fun load(path: Path): LoadedModel {
        val bytes = readPayload(path = path, accepted = listOf(VERSION_V1, VERSION))
        val dto = ProtoBuf.decodeFromByteArray(deserializer = SkeinModelDto.serializer(), bytes = bytes)
        return LoadedModel(
            schema = schemaOf(fields = dto.fields),
            classifier = ClassifierKindEnum.entries[dto.classifier],
            hashingConfig = HashingConfig(
                key0 = dto.key0,
                key1 = dto.key1,
                numFeatures = dto.numFeatures,
                charNgramMin = dto.charNgramMin,
                charNgramMax = dto.charNgramMax,
                wordNgramMin = dto.wordNgramMin,
                wordNgramMax = dto.wordNgramMax,
            ),
            observations = dto.observations.map { obs ->
                LabeledFeatures(
                    label = Label(value = obs.label),
                    features = FeatureVector(indices = obs.indices, values = obs.values),
                )
            },
            calibration = Calibration(temperature = dto.calibrationTemperature),
            hyperparameters = ClassifierHyperparameters(
                smoothingAlpha = dto.smoothingAlpha,
                initialLearningRate = dto.initialLearningRate,
                decayRate = dto.decayRate,
                l2Regularization = dto.l2Regularization,
            ),
        )
    }

    /**
     * Writes a fitted multi-label model.
     *
     * Stores the **weights**, not a corpus. The single-label format persists observations and
     * replays them through a fresh classifier on load, which works because those classifiers learn
     * one observation at a time. A batch-fitted model has no such history: replaying would mean
     * re-running L-BFGS over the whole corpus, turning a file open into a training run, and the
     * corpus would have to travel with the model to make it possible at all.
     *
     * [vectorizer] is recorded by fingerprint so [loadMultiLabel] can refuse a mismatch. Pass
     * [hashingConfig] when the vectorizer is this library's feature hashing and the settings are
     * worth carrying for inspection; it is not used to reconstruct anything, since the caller
     * supplies the vectorizer on load.
     *
     * A fitted [IdfVectorizer] is the one exception to that: its document-frequency table changes
     * every feature value, cannot be recovered from the corpus after the fact, and is therefore
     * written into the file automatically. Read it back with the [loadMultiLabel] overload taking a
     * factory.
     */
    fun saveMultiLabel(
        path: Path,
        schema: Schema,
        model: MultiLabelLogisticClassifier,
        vectorizer: Vectorizer,
        thresholds: LabelThresholds = LabelThresholds.uniform(threshold = DEFAULT_THRESHOLD),
        hashingConfig: HashingConfig? = null,
    ) {
        val weights = model.weights()
        val encoding = WeightCodec.narrowestEncoding(weights = weights.weights)
        val fingerprint = vectorizer.fingerprint()
        val canaryFields = canaryFields(canary = capturedCanary(vectorizer = vectorizer))
        val documentFrequencies = (vectorizer as? IdfVectorizer)?.table()
        val tuned = thresholds.asMap().entries.sortedBy { entry -> entry.key.value }
        val dto = MultiLabelModelDto(
            fields = schema.fields.map { field -> fieldDto(field = field) },
            vectorizerKind = fingerprint.kind,
            vectorizerDimension = fingerprint.dimension,
            vectorizerDigest = fingerprint.configDigest,
            labels = weights.labels.map { label -> label.value },
            intercepts = weights.intercepts,
            featureCount = weights.featureCount,
            featureOffsets = weights.featureOffsets,
            labelIndexDeltas = WeightCodec.encodeLabelIndices(
                featureOffsets = weights.featureOffsets,
                labelIndices = weights.labelIndices,
            ),
            weightEncoding = encoding.ordinal,
            weights = WeightCodec.encodeWeights(weights = weights.weights, encoding = encoding),
            l2Regularization = model.hyperparameters().l2Regularization,
            thresholdFallback = thresholds.fallback,
            tunedThresholdLabels = tuned.map { entry -> entry.key.value },
            tunedThresholds = tuned.map { entry -> entry.value }.toDoubleArray(),
            hasHashingConfig = hashingConfig != null,
            key0 = hashingConfig?.key0 ?: 0L,
            key1 = hashingConfig?.key1 ?: 0L,
            numFeatures = hashingConfig?.numFeatures ?: 0,
            charNgramMin = hashingConfig?.charNgramMin ?: 0,
            charNgramMax = hashingConfig?.charNgramMax ?: 0,
            wordNgramMin = hashingConfig?.wordNgramMin ?: 0,
            wordNgramMax = hashingConfig?.wordNgramMax ?: 0,
            termWeighting = hashingConfig?.termWeighting?.ordinal ?: 0,
            hasDocumentFrequencies = documentFrequencies != null,
            documentCount = documentFrequencies?.documentCount ?: 0,
            minimumDocumentFrequency = documentFrequencies?.minimumDocumentFrequency ?: 1,
            documentFrequencies = documentFrequencies?.frequencies ?: IntArray(size = 0),
            hasCanary = canaryFields.present,
            canaryProbes = canaryFields.probes,
            canaryVectors = canaryFields.vectors,
            canaryTolerance = canaryFields.tolerance,
        )
        val encoded = ProtoBuf.encodeToByteArray(serializer = MultiLabelModelDto.serializer(), value = dto)
        path.outputStream().use { file ->
            file.write(MAGIC)
            file.write(VERSION_MULTI_LABEL.toInt())
            GZIPOutputStream(file).use { gzip -> gzip.write(encoded) }
        }
    }

    /**
     * Reads a multi-label model, building the vectorizer from what the file itself carries.
     *
     * [vectorizerFactory] receives the stored [DocumentFrequencyTable], or `null` when the model was
     * trained without one, and returns the vectorizer to score with. This is the overload to use
     * with [IdfVectorizer], whose fitted table is in the file and cannot be reconstructed from
     * anything the caller holds. The fingerprint is still verified afterwards, so a factory that
     * ignores the table is caught rather than trusted.
     */
    fun loadMultiLabel(
        path: Path,
        vectorizerFactory: (DocumentFrequencyTable?) -> Vectorizer,
        verifyCanary: Boolean = true,
    ): LoadedMultiLabelModel {
        val dto = readMultiLabelDto(path = path)
        return decodeMultiLabel(
            dto = dto,
            vectorizer = vectorizerFactory(documentFrequencyTable(dto = dto)),
            verifyCanary = verifyCanary,
        )
    }

    /**
     * Reads a multi-label model, **refusing** it if [vectorizer] is not the one it was trained with.
     *
     * The check is the reason this takes a vectorizer at all. Scoring a model with a different
     * featurisation does not throw on its own and does not look wrong in the output — it simply
     * returns confident, incorrect labels for as long as nobody notices. See
     * [VectorizerMismatchException].
     *
     * **This can perform I/O beyond reading the file.** When the model carries a
     * [io.skein.classify.domain.VectorizerCanary], its probe texts are re-embedded through
     * [vectorizer] and compared, which for an embedding *service* means one network round trip per
     * probe — so opening a file can now block, or fail because the service is down. That is the
     * trade the canary buys: a loud failure at load instead of a silent one at inference. Pass
     * `verifyCanary = false` to skip it, and see [VectorizerCanaryException] for when to.
     *
     * A model saved without a canary is unaffected, and this parameter does nothing.
     */
    fun loadMultiLabel(
        path: Path,
        vectorizer: Vectorizer,
        verifyCanary: Boolean = true,
    ): LoadedMultiLabelModel {
        return decodeMultiLabel(
            dto = readMultiLabelDto(path = path),
            vectorizer = vectorizer,
            verifyCanary = verifyCanary,
        )
    }

    private fun readMultiLabelDto(path: Path): MultiLabelModelDto {
        val bytes = readPayload(path = path, accepted = listOf(VERSION_MULTI_LABEL))
        return ProtoBuf.decodeFromByteArray(deserializer = MultiLabelModelDto.serializer(), bytes = bytes)
    }

    /**
     * The vectorizer's canary, re-checked against the vectorizer **now**, at save time.
     *
     * The second check is the point. A canary captured before training proves what the vectorizer
     * produced then; capturing again here proves it still produces the same thing after the whole
     * corpus has gone through it. Without this, a model swapped on the server *during* a long
     * training run leaves half the corpus embedded by one model and half by another, and a canary
     * taken before the run still matches at load. The result is a model that is quietly garbage and
     * passes every check.
     */
    private fun capturedCanary(vectorizer: Vectorizer): VectorizerCanary? {
        val canary = vectorizer.canary() ?: return null
        verifyCanary(canary = canary, vectorizer = vectorizer)
        return canary
    }

    /** Re-embeds a canary's probes through [vectorizer] and throws if they have moved. */
    private fun verifyCanary(canary: VectorizerCanary, vectorizer: Vectorizer) {
        val observed = canary.probes.map { probe ->
            dense(vector = vectorizer.vectorize(text = probe), width = canary.dimension())
        }
        val worst = canary.worstDrift(observed = observed)
        if (worst.distance > canary.tolerance) {
            throw VectorizerCanaryException(canary = canary, probe = worst.probe, drift = worst.distance)
        }
    }

    /** The canary flattened into the shape the DTO stores, or the empty defaults when absent. */
    private fun canaryFields(canary: VectorizerCanary?): CanaryFields {
        return CanaryFields(
            present = canary != null,
            probes = canary?.probes.orEmpty(),
            vectors = flatten(vectors = canary?.references.orEmpty()),
            tolerance = canary?.tolerance ?: VectorizerCanary.DEFAULT_TOLERANCE,
        )
    }

    /**
     * Widens a [FeatureVector] to a dense array so a sparse vectorizer can carry a canary too.
     *
     * An embedding vectorizer already emits a dense run of indices, but nothing in the port
     * promises that, and reading [FeatureVector.values] directly would silently compare the wrong
     * positions for anything that does not.
     */
    private fun dense(vector: FeatureVector, width: Int): FloatArray {
        val dense = FloatArray(size = width)
        for (position in vector.indices.indices) {
            val index = vector.indices[position]
            if (index in 0 until width) {
                dense[index] = vector.values[position]
            }
        }
        return dense
    }

    private fun flatten(vectors: List<FloatArray>): FloatArray {
        if (vectors.isEmpty()) {
            return FloatArray(size = 0)
        }
        val flat = FloatArray(size = vectors.size * vectors.first().size)
        vectors.forEachIndexed { row, vector ->
            vector.copyInto(destination = flat, destinationOffset = row * vector.size)
        }
        return flat
    }

    private fun storedCanary(dto: MultiLabelModelDto): VectorizerCanary? {
        if (!dto.hasCanary) {
            return null
        }
        val width = dto.vectorizerDimension
        return VectorizerCanary(
            probes = dto.canaryProbes,
            references = dto.canaryProbes.indices.map { row ->
                dto.canaryVectors.copyOfRange(fromIndex = row * width, toIndex = (row + 1) * width)
            },
            tolerance = dto.canaryTolerance,
        )
    }

    private fun documentFrequencyTable(dto: MultiLabelModelDto): DocumentFrequencyTable? {
        if (!dto.hasDocumentFrequencies) {
            return null
        }
        return DocumentFrequencyTable(
            documentCount = dto.documentCount,
            frequencies = dto.documentFrequencies,
            minimumDocumentFrequency = dto.minimumDocumentFrequency,
        )
    }

    private fun decodeMultiLabel(
        dto: MultiLabelModelDto,
        vectorizer: Vectorizer,
        verifyCanary: Boolean,
    ): LoadedMultiLabelModel {
        val stored = VectorizerFingerprint(
            kind = dto.vectorizerKind,
            dimension = dto.vectorizerDimension,
            configDigest = dto.vectorizerDigest,
        )
        val supplied = vectorizer.fingerprint()
        if (stored != supplied) {
            throw VectorizerMismatchException(expected = stored, actual = supplied)
        }
        // Fingerprint first, canary second. The fingerprint is free; the canary costs one call
        // per probe, which for a remote service is a network round trip. A model loaded with
        // outright the wrong vectorizer must not pay for those before failing.
        val canary = storedCanary(dto = dto)
        if (canary != null && verifyCanary) {
            verifyCanary(canary = canary, vectorizer = vectorizer)
        }
        val encoding = WeightEncodingEnum.entries[dto.weightEncoding]
        val weights = MultiLabelWeights(
            labels = dto.labels.map { value -> Label(value = value) },
            intercepts = dto.intercepts,
            featureOffsets = dto.featureOffsets,
            labelIndices = WeightCodec.decodeLabelIndices(
                featureOffsets = dto.featureOffsets,
                deltas = dto.labelIndexDeltas,
            ),
            weights = WeightCodec.decodeWeights(bytes = dto.weights, encoding = encoding),
        )
        return LoadedMultiLabelModel(
            schema = schemaOf(fields = dto.fields),
            classifier = MultiLabelLogisticClassifier(
                weights = weights,
                tuning = ClassifierHyperparameters(l2Regularization = dto.l2Regularization),
            ),
            fingerprint = stored,
            canary = canary,
            thresholds = LabelThresholds(
                byLabel = dto.tunedThresholdLabels.indices.associate { index ->
                    Label(value = dto.tunedThresholdLabels[index]) to dto.tunedThresholds[index]
                },
                fallback = dto.thresholdFallback,
            ),
            hashingConfig = if (!dto.hasHashingConfig) {
                null
            } else {
                HashingConfig(
                    key0 = dto.key0,
                    key1 = dto.key1,
                    numFeatures = dto.numFeatures,
                    charNgramMin = dto.charNgramMin,
                    charNgramMax = dto.charNgramMax,
                    wordNgramMin = dto.wordNgramMin,
                    wordNgramMax = dto.wordNgramMax,
                    termWeighting = TermWeightingEnum.entries[dto.termWeighting],
                )
            },
        )
    }

    private fun fieldDto(field: FieldSpec): FieldDto {
        return FieldDto(
            type = when (field) {
                is TextField -> FIELD_TEXT
                is CategoricalField -> FIELD_CATEGORICAL
                is NumericField -> FIELD_NUMERIC
                is IdentifierField -> FIELD_IDENTIFIER
                is LabelField -> FIELD_LABEL
            },
            name = field.name,
            sensitivity = field.sensitivity.ordinal,
        )
    }

    private fun schemaOf(fields: List<FieldDto>): Schema {
        val builder = SchemaBuilder()
        fields.forEach { field ->
            val sensitivity = SensitivityEnum.entries[field.sensitivity]
            when (field.type) {
                FIELD_TEXT -> builder.text(name = field.name, sensitivity = sensitivity)
                FIELD_CATEGORICAL -> builder.categorical(name = field.name, sensitivity = sensitivity)
                FIELD_NUMERIC -> builder.numeric(name = field.name, sensitivity = sensitivity)
                FIELD_IDENTIFIER -> builder.identifier(name = field.name, sensitivity = sensitivity)
                FIELD_LABEL -> builder.label(name = field.name)
                else -> throw IllegalArgumentException("unknown field type ${field.type} in .skein file")
            }
        }
        return builder.build()
    }
}
