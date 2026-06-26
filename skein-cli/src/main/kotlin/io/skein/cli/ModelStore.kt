package io.skein.cli

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.FieldSpec
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
import java.nio.file.Path
import kotlin.io.path.readText
import kotlin.io.path.writeText

/**
 * Saves and loads a trained model as a compact, dependency-free text file.
 *
 * The file holds the fixed hashing key, the schema, the classifier kind and every labeled feature
 * vector. The model state is reproduced by *replaying* those observations into a fresh classifier
 * (`ClassificationService.retrain`) — Naive Bayes counts are reproduced exactly and Logistic
 * Regression weights deterministically — so nothing classifier-internal needs serializing.
 *
 * Privacy: the file contains the hashing key (the keyed-PRF secret) and the already-irreversible
 * hashed feature vectors — never clear-text record content. Treat it as a secret because of the key.
 *
 * ponytail: line-based format, no JSON/serialization dependency. Revisit if the schema or hashing
 * config grows. Labels containing a tab are not supported (tab separates label from features).
 */
object ModelStore {

    private const val MAGIC = "skein-model"
    private const val VERSION = "1"
    private const val FEATURE_SEPARATOR = ':'
    private const val LABEL_SEPARATOR = '\t'
    private const val FIELD_DEFINITION_PARTS = 4

    fun save(
        path: Path,
        schema: Schema,
        classifier: ClassifierKindEnum,
        hashingConfig: HashingConfig,
        observations: List<LabeledFeatures>,
    ) {
        val lines = ArrayList<String>(observations.size + schema.fields.size + 3)
        lines.add("$MAGIC $VERSION")
        lines.add("classifier ${classifier.name}")
        lines.add(hashingLine(config = hashingConfig))
        schema.fields.forEach { field -> lines.add(fieldLine(field = field)) }
        lines.add("---")
        observations.forEach { observation -> lines.add(observationLine(observation = observation)) }
        path.writeText(text = lines.joinToString(separator = "\n", postfix = "\n"))
    }

    fun load(path: Path): LoadedModel {
        val lines = path.readText().split("\n").filter { line -> line.isNotEmpty() }
        require(value = lines.isNotEmpty() && lines.first() == "$MAGIC $VERSION") {
            "malformed model file: missing or unsupported header"
        }
        val separator = lines.indexOf(element = "---")
        require(value = separator >= 0) { "malformed model file: missing '---' separator" }

        var classifier = ClassifierKindEnum.NAIVE_BAYES
        var hashingConfig: HashingConfig? = null
        val builder = SchemaBuilder()
        for (line in lines.subList(fromIndex = 1, toIndex = separator)) {
            when {
                line.startsWith(prefix = "classifier ") -> classifier = parseClassifier(line = line)
                line.startsWith(prefix = "hashing ") -> hashingConfig = parseHashing(line = line)
                line.startsWith(prefix = "field ") -> addField(builder = builder, line = line)
            }
        }
        require(value = hashingConfig != null) { "malformed model file: missing hashing config" }

        val observations = lines.subList(fromIndex = separator + 1, toIndex = lines.size)
            .map { line -> parseObservation(line = line) }
        return LoadedModel(
            schema = builder.build(),
            classifier = classifier,
            hashingConfig = hashingConfig,
            observations = observations,
        )
    }

    private fun hashingLine(config: HashingConfig): String {
        return "hashing ${config.key0} ${config.key1} ${config.numFeatures} " +
            "${config.charNgramMin} ${config.charNgramMax} ${config.wordNgramMin} ${config.wordNgramMax}"
    }

    private fun parseHashing(line: String): HashingConfig {
        val parts = line.removePrefix(prefix = "hashing ").trim().split(" ")
        require(value = parts.size == 7) { "malformed model file: bad hashing line" }
        return HashingConfig(
            key0 = parts[0].toLong(),
            key1 = parts[1].toLong(),
            numFeatures = parts[2].toInt(),
            charNgramMin = parts[3].toInt(),
            charNgramMax = parts[4].toInt(),
            wordNgramMin = parts[5].toInt(),
            wordNgramMax = parts[6].toInt(),
        )
    }

    private fun parseClassifier(line: String): ClassifierKindEnum {
        return ClassifierKindEnum.valueOf(value = line.removePrefix(prefix = "classifier ").trim())
    }

    private fun fieldLine(field: FieldSpec): String {
        val type = when (field) {
            is TextField -> "TEXT"
            is CategoricalField -> "CATEGORICAL"
            is NumericField -> "NUMERIC"
            is IdentifierField -> "IDENTIFIER"
            is LabelField -> "LABEL"
        }
        return "field $type ${field.name} ${field.sensitivity.name}"
    }

    private fun addField(builder: SchemaBuilder, line: String) {
        val parts = line.split(" ")
        require(value = parts.size == FIELD_DEFINITION_PARTS) { "malformed model file: bad field line '$line'" }
        val type = parts[1]
        val name = parts[2]
        val sensitivity = SensitivityEnum.valueOf(value = parts[3])
        when (type) {
            "TEXT" -> builder.text(name = name, sensitivity = sensitivity)
            "CATEGORICAL" -> builder.categorical(name = name, sensitivity = sensitivity)
            "NUMERIC" -> builder.numeric(name = name, sensitivity = sensitivity)
            "IDENTIFIER" -> builder.identifier(name = name, sensitivity = sensitivity)
            "LABEL" -> builder.label(name = name)
            else -> throw IllegalArgumentException("malformed model file: unknown field type '$type'")
        }
    }

    private fun observationLine(observation: LabeledFeatures): String {
        val features = observation.features
        val tokens = StringBuilder()
        for (position in features.indices.indices) {
            if (position > 0) {
                tokens.append(' ')
            }
            tokens.append(features.indices[position]).append(FEATURE_SEPARATOR).append(features.values[position])
        }
        return "${observation.label.value}$LABEL_SEPARATOR$tokens"
    }

    private fun parseObservation(line: String): LabeledFeatures {
        val tab = line.indexOf(char = LABEL_SEPARATOR)
        require(value = tab >= 0) { "malformed model file: observation without a label separator" }
        val label = Label(value = line.substring(startIndex = 0, endIndex = tab))
        val tokens = line.substring(startIndex = tab + 1).trim()
        if (tokens.isEmpty()) {
            val empty = FeatureVector(indices = IntArray(size = 0), values = FloatArray(size = 0))
            return LabeledFeatures(label = label, features = empty)
        }
        val pairs = tokens.split(" ")
        val indices = IntArray(size = pairs.size)
        val values = FloatArray(size = pairs.size)
        pairs.forEachIndexed { position, pair ->
            val colon = pair.indexOf(char = FEATURE_SEPARATOR)
            require(value = colon >= 0) { "malformed model file: bad feature token '$pair'" }
            indices[position] = pair.substring(startIndex = 0, endIndex = colon).toInt()
            values[position] = pair.substring(startIndex = colon + 1).toFloat()
        }
        return LabeledFeatures(label = label, features = FeatureVector(indices = indices, values = values))
    }
}
