package io.skein.classify.application

import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.FieldSpec
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.TextField

/**
 * Infers a [Schema] from a sample of records and a known label field name.
 *
 * Per non-label field, looking only at non-null sample values:
 * - all parse as numbers -> [NumericField]
 * - values repeat and stay within [maxCategoricalCardinality] distinct values -> [CategoricalField]
 * - all distinct and code-like (alphanumeric with a digit) -> [IdentifierField] (PII)
 * - otherwise (all values distinct, free text) -> [TextField]
 *
 * A simple, deterministic heuristic; for anything it cannot tell apart (e.g. all-digit identifiers,
 * which read as numeric) declare the field explicitly via the [Schema.define] DSL.
 */
class SchemaInference(private val maxCategoricalCardinality: Int = DEFAULT_MAX_CATEGORICAL_CARDINALITY) {

    fun infer(records: List<Record>, labelField: String): Schema {
        require(value = records.isNotEmpty()) { "cannot infer a schema from zero records" }
        val fieldNames = collectFieldNames(records = records)
        val specs = ArrayList<FieldSpec>()
        for (name in fieldNames) {
            if (name == labelField) {
                specs.add(LabelField(name = name))
            } else {
                specs.add(inferField(name = name, values = valuesOf(records, name)))
            }
        }
        if (fieldNames.none { name -> name == labelField }) {
            specs.add(LabelField(name = labelField))
        }
        return Schema(fields = specs)
    }

    private fun collectFieldNames(records: List<Record>): Set<String> {
        val names = LinkedHashSet<String>()
        records.forEach { record -> names.addAll(record.fieldNames()) }
        return names
    }

    private fun valuesOf(records: List<Record>, name: String): List<String> {
        return records.mapNotNull { record -> record[name]?.toString()?.trim() }
            .filter { value -> value.isNotEmpty() }
    }

    private fun inferField(name: String, values: List<String>): FieldSpec {
        if (values.isNotEmpty() && values.all { value -> value.toDoubleOrNull() != null }) {
            return NumericField(name = name)
        }
        // Categorical requires repetition: at least one value recurs and the distinct set stays small.
        // Fields where every sampled value is unique are treated as free text.
        val distinct = values.toSet().size
        if (distinct in 1 until values.size && distinct <= maxCategoricalCardinality) {
            return CategoricalField(name = name)
        }
        if (looksLikeIdentifier(values)) {
            return IdentifierField(name = name)
        }
        return TextField(name = name)
    }

    /**
     * Heuristic: every value is distinct (high cardinality) and looks like a code — a space-free
     * alphanumeric token containing at least one digit and at least [MIN_IDENTIFIER_LENGTH] chars
     * (e.g. `AB12345`, `DE89370400440532013000`). Such fields default to PII [IdentifierField].
     */
    private fun looksLikeIdentifier(values: List<String>): Boolean {
        if (values.isEmpty() || values.toSet().size != values.size) {
            return false
        }
        return values.all { value ->
            value.length >= MIN_IDENTIFIER_LENGTH &&
                IDENTIFIER.matches(value) &&
                value.any { character -> character.isDigit() }
        }
    }

    companion object {
        const val DEFAULT_MAX_CATEGORICAL_CARDINALITY = 20
        private const val MIN_IDENTIFIER_LENGTH = 4
        private val IDENTIFIER = Regex("[\\p{L}\\d]+")
    }
}
