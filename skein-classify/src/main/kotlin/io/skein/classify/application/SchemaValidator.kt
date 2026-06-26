package io.skein.classify.application

import io.skein.classify.domain.NumericField
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.ValidationResult

/**
 * Validates a [Record] against a [Schema].
 *
 * Errors (reject the record): the label field is missing or blank; a numeric field holds a
 * non-numeric value. Warnings (accepted with note): a declared non-label field is absent, or the
 * record carries a field the schema does not declare.
 */
class SchemaValidator(private val schema: Schema) {

    fun validate(record: Record): ValidationResult {
        val errors = mutableListOf<String>()
        val warnings = mutableListOf<String>()

        validateLabel(record = record, errors = errors)
        validateDeclaredFields(record = record, errors = errors, warnings = warnings)
        warnUnknownFields(record = record, warnings = warnings)

        return ValidationResult(errors = errors, warnings = warnings)
    }

    private fun validateLabel(record: Record, errors: MutableList<String>) {
        val labelName = schema.labelField.name
        val raw = record[labelName]
        if (raw == null || raw.toString().isBlank()) {
            errors.add("missing or blank label field '$labelName'")
        }
    }

    private fun validateDeclaredFields(
        record: Record,
        errors: MutableList<String>,
        warnings: MutableList<String>,
    ) {
        for (field in schema.fields.filter { spec -> spec != schema.labelField }) {
            val raw = record[field.name]
            when {
                raw == null -> warnings.add("missing field '${field.name}'")
                field is NumericField && !isNumeric(value = raw) ->
                    errors.add("field '${field.name}' must be numeric but was '$raw'")
            }
        }
    }

    private fun warnUnknownFields(record: Record, warnings: MutableList<String>) {
        for (name in record.fieldNames()) {
            if (schema.field(name = name) == null) {
                warnings.add("unknown field '$name' is not declared in the schema")
            }
        }
    }

    private fun isNumeric(value: Any?): Boolean {
        return when (value) {
            is Number -> true
            else -> value.toString().trim().toDoubleOrNull() != null
        }
    }
}
