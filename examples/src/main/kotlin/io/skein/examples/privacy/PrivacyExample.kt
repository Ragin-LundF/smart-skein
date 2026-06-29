package io.skein.examples.privacy

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.RecordMapper
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.SensitivityEnum

fun runPrivacyExample() {
    println("=== Privacy Example ===")
    println()

    // Schema: 'iban' is PII → excluded from features. 'purpose' is PUBLIC → included.
    val schema = Schema.define {
        text(name = "purpose")
        identifier(name = "iban", sensitivity = SensitivityEnum.PII)
        label(name = "type")
    }

    // RecordMapper demonstrates exactly which fields enter the feature text.
    val mapper = RecordMapper(schema = schema)

    println("Field sensitivity in this schema:")
    schema.fields.forEach { field ->
        println("  %-10s sensitivity=%s".format(field.name, field.sensitivity))
    }
    println()

    val record = Record(
        values = mapOf(
            "purpose" to "rent payment monthly apartment",
            "iban" to "DE89370400440532013000",
            "type" to "rent",
        ),
    )

    val mapped = mapper.map(record = record)
    println("Record fields    : ${record.fieldNames().sorted()}")
    println("MappedRecord.featureText : \"${mapped.featureText}\"")
    println("  → 'iban' (PII) is absent from featureText; 'purpose' (PUBLIC) is present.")
    println("  → 'type' (LabelField) is never a feature — it is the target.")
    println()

    // FEATURES_ONLY mode — the default for production.
    val serviceFeatures = ClassificationService(
        schema = schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig(key0 = 0x707269766163L, key1 = 0x796578616d70L),
    )

    fun rec(purpose: String, iban: String, type: String): Record =
        Record(values = mapOf("purpose" to purpose, "iban" to iban, "type" to type))

    val trainingRecords = listOf(
        rec(purpose = "rent payment monthly", iban = "DE89370400440532013000", type = "rent"),
        rec(purpose = "monthly rent apartment", iban = "DE44500105175407324931", type = "rent"),
        rec(purpose = "salary october payout employer", iban = "DE12345678901234567890", type = "salary"),
        rec(purpose = "monthly salary payment", iban = "DE99887766554433221100", type = "salary"),
    )
    serviceFeatures.learnAll(records = trainingRecords)

    val probe = Record(
        values = mapOf("purpose" to "rent apartment payment", "iban" to "DE00000000000000000000"),
    )
    val prediction = serviceFeatures.classify(record = probe)
    println("FEATURES_ONLY mode classification:")
    println("  privacy mode : ${serviceFeatures.privacyMode}")
    println("  purpose      : \"${probe["purpose"]}\"")
    println("  iban         : ${probe["iban"]}  ← not used for features")
    println("  predicted    : ${prediction.label.value} (confidence ${"%.2f".format(prediction.confidence)})")
    println()

    printPrivacyGuarantee()
}

private fun printPrivacyGuarantee() {
    println("ENCRYPTED_SOURCE mode:")
    println("  privacyMode = PrivacyModeEnum.ENCRYPTED_SOURCE")
    println("  The source record is retained encrypted alongside the feature vector.")
    println("  This allows later re-mapping (e.g. schema migration, audit).")
    println("  Requires a store that supports encryption — use skein-store-postgres.")
    println("  InMemoryFeatureStore ignores the distinction: both modes store only feature vectors.")
    println()
    println("Privacy guarantee in both modes:")
    println("  PII fields (sensitivity = SensitivityEnum.PII) never enter the feature text.")
    println("  Feature hashing is a keyed PRF: the hashing key is the privacy secret.")
    println("  Without the key, hashed feature indices cannot be reversed to the source text.")
}
