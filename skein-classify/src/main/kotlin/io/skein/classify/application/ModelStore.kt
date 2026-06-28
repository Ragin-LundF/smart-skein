@file:OptIn(ExperimentalSerializationApi::class)

package io.skein.classify.application

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SchemaBuilder
import io.skein.classify.domain.SensitivityEnum
import io.skein.classify.domain.TextField
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.Serializable
import kotlinx.serialization.protobuf.ProtoBuf
import java.nio.file.Path
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlin.io.path.inputStream
import kotlin.io.path.outputStream

// Layout: 4-byte magic 'SKEI' + 1-byte version, then GZIP(ProtoBuf payload).
// Privacy: the hashing key (keyed-PRF secret) and already-irreversible feature vectors are stored — treat as a secret.
private val MAGIC = byteArrayOf(0x53, 0x4B, 0x45, 0x49) // 'SKEI'
private const val VERSION: Byte = 0x01

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
)

@Serializable
private data class FieldDto(val type: Int, val name: String, val sensitivity: Int)

@Serializable
private class ObservationDto(val label: String, val indices: IntArray, val values: FloatArray)

object ModelStore {

    fun save(
        path: Path,
        schema: Schema,
        classifier: ClassifierKindEnum,
        hashingConfig: HashingConfig,
        observations: List<LabeledFeatures>,
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
            fields = schema.fields.map { field ->
                FieldDto(
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
            },
            observations = observations.map { obs ->
                ObservationDto(
                    label = obs.label.value,
                    indices = obs.features.indices,
                    values = obs.features.values,
                )
            },
        )
        val encoded = ProtoBuf.encodeToByteArray(serializer = SkeinModelDto.serializer(), value = dto)
        path.outputStream().use { file ->
            file.write(MAGIC)
            file.write(VERSION.toInt())
            GZIPOutputStream(file).use { gzip -> gzip.write(encoded) }
        }
    }

    fun load(path: Path): LoadedModel {
        val bytes = path.inputStream().use { file ->
            val header = file.readNBytes(5)
            require(
                value = header.size == 5 &&
                    MAGIC.indices.all { i -> header[i] == MAGIC[i] } &&
                    header[4] == VERSION,
            ) { "malformed or unsupported .skein file (expected SKEI binary format v1)" }
            GZIPInputStream(file).readBytes()
        }
        val dto = ProtoBuf.decodeFromByteArray(deserializer = SkeinModelDto.serializer(), bytes = bytes)
        val builder = SchemaBuilder()
        dto.fields.forEach { field ->
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
        return LoadedModel(
            schema = builder.build(),
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
        )
    }
}
