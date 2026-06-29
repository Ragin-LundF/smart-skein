package io.skein.examples.importing

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.SchemaValidator
import io.skein.classify.application.RecordImportService
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.spi.RecordSource

private val SCHEMA = Schema.define {
    text(name = "purpose")
    numeric(name = "amount")
    identifier(name = "iban")
    label(name = "type")
}

private fun row(
    purpose: String, amount: String, iban: String, type: String? = null, note: String? = null,
): Map<String, Any?> {
    val m = mutableMapOf<String, Any?>("purpose" to purpose, "amount" to amount, "iban" to iban)
    if (type != null) m["type"] = type
    if (note != null) m["note"] = note
    return m
}

private val RAW_ROWS: List<Map<String, Any?>> = listOf(
    row(purpose = "rent payment monthly",    amount = "850.00",  iban = "DE89370400440532013000", type = "rent"),
    row(purpose = "salary october payout",   amount = "2400.00", iban = "DE44500105175407324931", type = "salary"),
    row(purpose = "insurance premium AIG",   amount = "67.89",   iban = "DE12345678901234567890", type = "insurance"),
    row(purpose = "monthly rent apartment",  amount = "850.00",  iban = "DE99887766554433221100", type = "rent"),
    row(purpose = "unknown transaction",     amount = "10.00",   iban = "DE00112233445566778899"),
    row(purpose = "salary advance",          amount = "n/a",     iban = "DE11223344556677889900", type = "salary"),
    row(purpose = "insurance premium Geico", amount = "90.00", iban = "DE55667788990011223344",
        type = "insurance", note = "annual"),
)

fun runRecordImportExample() {
    println("=== Record Import Example ===")
    println()

    val source = object : RecordSource {
        override fun stream(): Sequence<Record> = RAW_ROWS.asSequence().map { row -> Record(values = row) }
    }

    val result = RecordImportService(schema = SCHEMA).importFrom(source = source)
    println("Import result:")
    println("  total submitted : ${RAW_ROWS.size}")
    println("  accepted        : ${result.acceptedCount()}")
    println("  rejected        : ${result.rejectedCount()}")
    println()

    if (result.rejected.isNotEmpty()) {
        println("Rejected records:")
        result.rejected.forEach { rejected ->
            println("  purpose: \"${rejected.record["purpose"] ?: "(no purpose)"}\"")
            rejected.reasons.forEach { reason -> println("    reason: $reason") }
        }
        println()
    }

    if (result.warnings.isNotEmpty()) {
        println("Warnings (accepted records with notes):")
        result.warnings.forEach { warning -> println("  $warning") }
        println()
    }

    val validator = SchemaValidator(schema = SCHEMA)
    val acceptedRecords = RAW_ROWS
        .map { row -> Record(values = row) }
        .filter { record -> validator.validate(record = record).isValid() }

    val service = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig(key0 = 0x696d706f7274L, key1 = 0x6578616d706cL),
    )
    service.learnAll(records = acceptedRecords)

    val metrics = service.metrics()
    println("ClassificationService learned from accepted records:")
    println("  total observations: ${metrics.totalObservations}")
    metrics.perLabelCounts.forEach { (label, count) -> println("  label '${label.value}': $count observation(s)") }
    println()

    val probe = Record(values = mapOf("purpose" to "rent monthly apartment", "iban" to "DE00000000000000000000"))
    val prediction = service.classify(record = probe)
    println("Classify probe \"rent monthly apartment\":")
    println("  predicted : ${prediction.label.value} (confidence ${"%.2f".format(prediction.confidence)})")
}
