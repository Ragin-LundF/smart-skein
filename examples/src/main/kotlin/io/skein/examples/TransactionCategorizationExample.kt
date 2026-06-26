package io.skein.examples

import io.skein.classify.application.ClassificationService
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.extract.application.SlotExtractor
import io.skein.extract.domain.GroupComponent
import io.skein.extract.domain.KeyAnchoredSlot
import io.skein.extract.domain.RepeatingGroupSlot
import io.skein.extract.domain.SlotDefinition
import io.skein.text.domain.TokenTypeEnum

/**
 * End-to-end Skein use case: **classify → route → extract** over bank-transaction purpose text.
 *
 * 1. A [ClassificationService] (Naive Bayes, privacy-preserving features) learns to categorize
 *    transactions from their purpose text.
 * 2. The predicted category routes the record to category-specific extraction rules.
 * 3. A [SlotExtractor] pulls the structured values that matter for that category.
 *
 * The model is trained on a few labeled samples when the example is constructed.
 */
class TransactionCategorizationExample {

    private val schema = Schema.define {
        text(name = "purpose")
        identifier(name = "iban")
        label(name = "category")
    }

    private val engine = ClassificationService(
        schema = schema,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig.randomKey(),
    )
    private val extractor = SlotExtractor()

    init {
        train()
    }

    /** Classifies [purpose] then extracts the fields routed for the predicted category. */
    fun process(purpose: String): PipelineResult {
        val prediction = engine.classify(Record(values = mapOf("purpose" to purpose, "iban" to "DE00")))
        val extraction = extractor.extract(text = purpose, slots = slotsFor(label = prediction.label))
        return PipelineResult(category = prediction.label, confidence = prediction.confidence, extracted = extraction)
    }

    private fun slotsFor(label: Label): List<SlotDefinition> {
        return when (label.value) {
            "insurance" -> listOf(
                RepeatingGroupSlot(
                    name = "policies",
                    components = listOf(
                        GroupComponent(name = "insurer", type = TokenTypeEnum.WORD_SYMBOL),
                        GroupComponent(name = "amount", type = TokenTypeEnum.AMOUNT),
                    ),
                ),
            )
            "rent" -> listOf(
                KeyAnchoredSlot(name = "customer", anchor = "CustomerNumber", targetType = TokenTypeEnum.ALPHANUMERIC),
            )
            else -> emptyList()
        }
    }

    private fun train() {
        engine.learnAll(
            listOf(
                sample(purpose = "AIG-Life 67.89 Geico-Auto 120.00 insurance premium", category = "insurance"),
                sample(purpose = "Allstate-Home 90.00 insurance premium", category = "insurance"),
                sample(purpose = "insurance premium annual policy", category = "insurance"),
                sample(purpose = "rent apartment CustomerNumber AB123 monthly", category = "rent"),
                sample(purpose = "monthly rent apartment CustomerNumber XY999", category = "rent"),
                sample(purpose = "rent apartment payment", category = "rent"),
                sample(purpose = "salary october payout", category = "salary"),
                sample(purpose = "monthly salary payment employer", category = "salary"),
            ),
        )
    }

    private fun sample(purpose: String, category: String): Record {
        return Record(values = mapOf("purpose" to purpose, "iban" to "DE00", "category" to category))
    }
}
