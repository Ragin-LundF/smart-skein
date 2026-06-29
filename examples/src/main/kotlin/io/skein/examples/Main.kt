package io.skein.examples

import io.skein.examples.activelearning.runActiveLearningExample
import io.skein.examples.cli.runCliDemoExample
import io.skein.examples.cli.runCliToolDemoExample
import io.skein.examples.clustering.runTemplateClustererExample
import io.skein.examples.crf.runCrfTaggerExample
import io.skein.examples.importing.runRecordImportExample
import io.skein.examples.logwatch.logWatchMain
import io.skein.examples.patternmatching.runTokenPatternExample
import io.skein.examples.persistence.runModelPersistenceExample
import io.skein.examples.privacy.runPrivacyExample
import io.skein.examples.regression.runLogisticRegressionExample
import io.skein.examples.schema.runSchemaInferenceExample
import io.skein.examples.slots.runPositionalSlotExample
import io.skein.examples.text.runNormalizerEdgeCasesExample
import io.skein.examples.text.runSignatureExample
import io.skein.examples.textrepair.runTextRepairExample
import io.skein.examples.tokenization.runCustomTokenPatternsExample
import io.skein.examples.tokenization.runTokenizationModesExample
import io.skein.examples.validation.runSchemaValidationExample

private val USAGE = """
    Skein examples — runnable demonstrations of the classify/extract pipeline.

    Usage: examples <use-case> [args...]

    Use cases (Priority 1 — foundations):
      transaction       Classify bank-transaction text → route → extract structured fields.
      textrepair        Train a FrequencyModel, repair broken words, full normalize→tokenize pipeline.
      patternmatching   TokenPattern DSL, findAll, matchesFully, partial and non-matching cases.
      slots             PositionalSlot vs KeyAnchoredSlot: fixed-index and key-anchored extraction.
      schemainference   Infer a schema from sample records, classify with it, show edge cases.
      validation        SchemaValidator: valid, invalid, and warning-bearing records.
      clustering        TemplateClusterer: unsupervised layout discovery from a mixed corpus.
      persistence       ModelStore save/load round-trip; FrequencyModel serialize/deserialize.
      import            RecordImportService: stream, validate, accept/reject, feed into classifier.

    Use cases (Priority 2 — intermediate workflows):
      regression        Naive Bayes → Logistic Regression retraining, confidence comparison.
      activelearning    ActiveLearningSelector: pick uncertain candidates, feedback, metrics.
      crf               CrfSequenceLabeler: train on token sequences, generalize to unseen input.
      clidemo           Schema inference + active learning loop with equivalent CLI command hints.
      tokenization      WHITESPACE vs PUNCTUATION_AWARE mode comparison.
      customtokens      Custom TokenPatternConfig with a domain order-code recognizer.

    Use cases (Priority 3 — edge cases):
      normalizeredges   DefaultTextNormalizer idempotence and boundary behaviour.
      privacy           PII field exclusion from features, FEATURES_ONLY mode.
      signature         PatternSignature: identical layouts → same fingerprint, different → distinct.
      clitool           Predict all rows + export model to human-readable text (library-mode demo).

    Use cases (logwatch):
      logwatch          Train an anomaly detector from a keyword rules CSV, then scan log files.
                        train [--rules rules.csv] [--model model.skein]
                        scan  --log app.log | --sample  [--out results.csv] [--confidence 0.5]
""".trimIndent()

// Dispatch table keeps main() under the CyclomaticComplexMethod threshold.
private val DISPATCH: Map<String, () -> Unit> = mapOf(
    "transaction" to ::runTransactionExample,
    "textrepair" to ::runTextRepairExample,
    "patternmatching" to ::runTokenPatternExample,
    "slots" to ::runPositionalSlotExample,
    "schemainference" to ::runSchemaInferenceExample,
    "validation" to ::runSchemaValidationExample,
    "clustering" to ::runTemplateClustererExample,
    "persistence" to ::runModelPersistenceExample,
    "import" to ::runRecordImportExample,
    "regression" to ::runLogisticRegressionExample,
    "activelearning" to ::runActiveLearningExample,
    "crf" to ::runCrfTaggerExample,
    "clidemo" to ::runCliDemoExample,
    "tokenization" to ::runTokenizationModesExample,
    "customtokens" to ::runCustomTokenPatternsExample,
    "normalizeredges" to ::runNormalizerEdgeCasesExample,
    "privacy" to ::runPrivacyExample,
    "signature" to ::runSignatureExample,
    "clitool" to ::runCliToolDemoExample,
)

fun main(args: Array<String>) {
    val command = args.firstOrNull()
    if (command == "logwatch") {
        logWatchMain(args = args.drop(1).toTypedArray())
        return
    }
    DISPATCH[command]?.invoke() ?: println(USAGE)
}

private fun runTransactionExample() {
    val example = TransactionCategorizationExample()
    val transactions = listOf(
        "AIG-Life 67.89 Geico-Auto 120.00 insurance premium",
        "rent apartment CustomerNumber CD456 monthly",
        "salary november payout",
    )
    transactions.forEach { purpose ->
        val result = example.process(purpose = purpose)
        println("purpose : $purpose")
        println("  category   : ${result.category.value} (confidence ${"%.2f".format(result.confidence)})")
        if (result.extracted.fields.isEmpty()) {
            println("  extracted  : (no fields routed for this category)")
        } else {
            result.extracted.fields.forEach { field ->
                println("  extracted  : ${field.name} = ${field.value}")
            }
        }
    }
}
