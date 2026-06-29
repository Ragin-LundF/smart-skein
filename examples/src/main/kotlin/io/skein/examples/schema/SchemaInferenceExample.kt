package io.skein.examples.schema

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.SchemaInference
import io.skein.classify.domain.CategoricalField
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.IdentifierField
import io.skein.classify.domain.LabelField
import io.skein.classify.domain.NumericField
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.TextField

fun runSchemaInferenceExample() {
    println("=== Schema Inference Example ===")
    println()

    // A realistic bank-transaction corpus with mixed field types.
    // 'iban' is code-like (alphanumeric with digits, all distinct) → should infer IdentifierField.
    // 'amount' parses as a double every time            → NumericField.
    // 'type' repeats within a small set of values       → CategoricalField.
    // 'purpose' is free-form text                       → TextField.
    fun row(amount: String, iban: String, purpose: String, type: String): Record =
        Record(values = mapOf("amount" to amount, "iban" to iban, "purpose" to purpose, "type" to type))

    val records = listOf(
        row(amount = "12.50",   iban = "DE89123400", purpose = "rent payment monthly",         type = "rent"),
        row(amount = "1200.00", iban = "DE44500105", purpose = "salary october payout",         type = "salary"),
        row(amount = "67.89",   iban = "DE12345678", purpose = "insurance premium AIG life",    type = "insurance"),
        row(amount = "90.00",   iban = "DE99876543", purpose = "monthly rent apartment",        type = "rent"),
        row(amount = "1200.00", iban = "DE11223344", purpose = "salary november payment",       type = "salary"),
        row(amount = "120.00",  iban = "DE55667788", purpose = "insurance premium Geico auto",  type = "insurance"),
    )

    val inference = SchemaInference(maxCategoricalCardinality = 20)
    val schema = inference.infer(records = records, labelField = "type")

    println("Inferred schema fields:")
    schema.fields.forEach { field ->
        val kind = when (field) {
            is NumericField     -> "NUMERIC      (all values parse as Double)"
            is CategoricalField -> "CATEGORICAL  (values repeat, cardinality ≤ 20)"
            is IdentifierField  -> "IDENTIFIER   (all distinct, code-like → defaults to PII)"
            is TextField        -> "TEXT         (free-form, no pattern fits)"
            is LabelField       -> "LABEL"
        }
        println("  %-10s → %s".format(field.name, kind))
    }
    println()

    // Edge-case explanation: a purely numeric IBAN (e.g. "123456789012") would be inferred as
    // NumericField rather than IdentifierField, because the numeric check runs first. For such
    // fields, declare the schema explicitly via Schema.define { identifier("iban") }.
    println("Edge-case note:")
    println("  A purely numeric ID like '123456789012' would be inferred as NUMERIC,")
    println("  not IDENTIFIER, because the numeric check takes priority.")
    println("  Declare ambiguous fields explicitly with Schema.define { identifier(\"iban\") }.")
    println()

    // Train and classify with the inferred schema to prove it is usable end-to-end.
    val service = ClassificationService(
        schema = schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig(key0 = 0x736b65696e736368L, key1 = 0x656d61696e666572L),
    )
    service.learnAll(records = records)

    val probe = Record(
        values = mapOf(
            "amount" to "55.00",
            "iban" to "DE00000000",
            "purpose" to "insurance premium annual renewal",
        ),
    )
    val prediction = service.classify(record = probe)
    println("Classification with inferred schema:")
    println("  purpose  : \"insurance premium annual renewal\"")
    println("  predicted: ${prediction.label.value} (confidence ${"%.2f".format(prediction.confidence)})")
    prediction.alternatives.forEach { alt ->
        println("  alt      : ${alt.label.value} p=${"%.3f".format(alt.probability)}")
    }
}
