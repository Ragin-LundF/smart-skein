package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.MappedRecord
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SensitivityEnum

/**
 * Reduces a [Record] to a [MappedRecord]: the concatenated text of feature-eligible fields plus
 * the extracted label.
 *
 * A field contributes to the feature text only when it is neither the label nor PII — PII fields
 * are excluded so personal data never enters the feature representation in clear text.
 */
class RecordMapper(private val schema: Schema) {

    fun map(record: Record): MappedRecord {
        val featureText = buildFeatureText(record = record)
        val label = extractLabel(record = record)
        return MappedRecord(featureText = featureText, label = label)
    }

    private fun buildFeatureText(record: Record): String {
        return schema.fields
            .filter { field -> field !is LabelField && field.sensitivity != SensitivityEnum.PII }
            .mapNotNull { field -> record[field.name]?.toString()?.trim()?.takeIf { text -> text.isNotEmpty() } }
            .joinToString(separator = " ")
    }

    private fun extractLabel(record: Record): Label? {
        val raw = record[schema.labelField.name]?.toString()?.trim()
        return if (raw.isNullOrBlank()) null else Label(value = raw)
    }
}
