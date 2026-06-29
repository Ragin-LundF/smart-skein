package io.skein.examples.validation

import io.skein.classify.application.SchemaValidator
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema

private val SCHEMA = Schema.define {
    text(name = "purpose")
    numeric(name = "amount")
    identifier(name = "iban")
    label(name = "type")
}

private fun rec(purpose: String, amount: String? = null, iban: String? = null, type: String? = null): Record =
    Record(values = buildMap {
        put("purpose", purpose)
        if (amount != null) put("amount", amount)
        if (iban != null) put("iban", iban)
        if (type != null) put("type", type)
    })

private val VALIDATION_CASES: List<Pair<String, Record>> = listOf(
    "valid record" to rec(
        purpose = "rent payment monthly", amount = "850.00", iban = "DE89370400440532013000", type = "rent",
    ),
    "missing label" to rec(purpose = "salary november", amount = "2400.00", iban = "DE44500105175407324931"),
    "non-numeric amount" to rec(
        purpose = "insurance premium", amount = "n/a", iban = "DE12345678901234567890", type = "insurance",
    ),
    "unknown field + missing schema field" to Record(
        values = mapOf("purpose" to "grocery shopping", "type" to "other", "extra_column" to "unexpected"),
    ),
)

fun runSchemaValidationExample() {
    println("=== Schema Validation Example ===")
    println()

    val validator = SchemaValidator(schema = SCHEMA)

    VALIDATION_CASES.forEach { (description, record) ->
        val result = validator.validate(record = record)
        println("[$description]")
        println("  valid    : ${result.isValid()}")
        if (result.errors.isNotEmpty()) {
            result.errors.forEach { error -> println("  error    : $error") }
        }
        if (result.warnings.isNotEmpty()) {
            result.warnings.forEach { warning -> println("  warning  : $warning") }
        }
        println()
    }

    // Batch summary — shows the pattern for bulk import pipelines.
    println("--- Batch validation summary ---")
    val allRecords = VALIDATION_CASES.map { (_, record) -> record }
    val accepted = allRecords.count { record -> validator.validate(record = record).isValid() }
    val rejected = allRecords.size - accepted
    println("  total    : ${allRecords.size}")
    println("  accepted : $accepted")
    println("  rejected : $rejected")
    println()
    println("Use RecordImportService to combine validation + mapping in one pipeline step.")
}
