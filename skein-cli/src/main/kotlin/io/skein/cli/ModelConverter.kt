package io.skein.cli

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FieldSpec
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.TextField
import java.nio.file.Path
import kotlin.io.path.writeText

/** Converts a binary .skein file to the human-readable text format for inspection. */
object ModelConverter {

    private const val FEATURE_SEPARATOR = ':'
    private const val LABEL_SEPARATOR = '\t'

    fun toText(src: Path, dst: Path) {
        val model = ModelStore.load(path = src)
        val lines = ArrayList<String>(model.observations.size + model.schema.fields.size + 3)
        lines.add("skein-model 1")
        lines.add("classifier ${model.classifier.name}")
        lines.add(hashingLine(model))
        model.schema.fields.forEach { field -> lines.add(fieldLine(field)) }
        lines.add("---")
        model.observations.forEach { obs -> lines.add(observationLine(obs)) }
        dst.writeText(text = lines.joinToString(separator = "\n", postfix = "\n"))
    }

    private fun hashingLine(model: LoadedModel): String {
        val c = model.hashingConfig
        return "hashing ${c.key0} ${c.key1} ${c.numFeatures} " +
            "${c.charNgramMin} ${c.charNgramMax} ${c.wordNgramMin} ${c.wordNgramMax}"
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

    private fun observationLine(obs: LabeledFeatures): String {
        val features = obs.features
        val tokens = StringBuilder()
        for (i in features.indices.indices) {
            if (i > 0) tokens.append(' ')
            tokens.append(features.indices[i]).append(FEATURE_SEPARATOR).append(features.values[i])
        }
        return "${obs.label.value}$LABEL_SEPARATOR$tokens"
    }
}
