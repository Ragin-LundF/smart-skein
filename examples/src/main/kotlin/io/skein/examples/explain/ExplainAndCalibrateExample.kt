package io.skein.examples.explain

import io.skein.classify.application.ClassificationService
import io.skein.classify.application.ModelEvaluator
import io.skein.classify.application.StratifiedSplitter
import io.skein.classify.domain.AttributionModeEnum
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import java.util.Locale

private const val HOLDOUT_RATIO = 0.3
private const val CONTRIBUTION_LIMIT = 6
private const val DIVIDER_WIDTH = 66
private const val ABSTAIN_THRESHOLD = 0.9

private val SCHEMA = Schema.define {
    text(name = "purpose")
    identifier(name = "iban")
    label(name = "category")
}

/**
 * Deliberately overlapping vocabulary. "monthly", "transfer" and "payment" appear under several
 * labels, and a few rows are genuinely ambiguous, so the model is confidently wrong often enough
 * for calibration to have something to correct. A perfectly separable toy corpus would be scored
 * perfectly on the holdout, and minimizing log-likelihood there simply sharpens the model further —
 * a correct outcome that demonstrates nothing.
 */
private val TRAINING = listOf(
    "monthly rent transfer to landlord" to "RENT",
    "rent payment for the apartment" to "RENT",
    "landlord standing order monthly" to "RENT",
    "apartment payment due monthly" to "RENT",
    "monthly transfer payment" to "RENT",
    "standing order transfer" to "RENT",
    "supermarket groceries weekly shop" to "GROCERIES",
    "groceries at the discount market" to "GROCERIES",
    "weekly supermarket food shop" to "GROCERIES",
    "market payment weekly" to "GROCERIES",
    "monthly market transfer" to "GROCERIES",
    "payment at the market" to "GROCERIES",
    "salary payment from employer" to "SALARY",
    "monthly salary transfer" to "SALARY",
    "employer payroll monthly" to "SALARY",
    "monthly transfer from employer" to "SALARY",
    "monthly payment transfer" to "SALARY",
    "transfer standing order monthly" to "SALARY",
)

private fun number(value: Double): String {
    return String.format(Locale.ROOT, "%.4f", value)
}

fun runExplainAndCalibrateExample() {
    println("=== Calibration and Explanations ===")
    println()

    val service = ClassificationService(
        schema = SCHEMA,
        privacyMode = PrivacyModeEnum.FEATURES_ONLY,
        hashingConfig = HashingConfig(key0 = 20260907L, key1 = 12L),
    )
    TRAINING.forEach { (purpose, category) ->
        service.learn(
            record = Record(values = mapOf("purpose" to purpose, "iban" to "DE0000", "category" to category)),
        )
    }
    println("Trained on ${service.metrics().totalObservations} observations.")
    println()

    val probe = Record(values = mapOf("purpose" to "rent transfer to my landlord", "iban" to "DE4242"))
    val split = StratifiedSplitter(seed = 7L).holdout(
        observations = service.featureStore.all(),
        testRatio = HOLDOUT_RATIO,
    )

    calibrate(service = service, probe = probe, holdout = split.holdout)
    abstain(service = service)
    explain(service = service, probe = probe)
    measure(service = service, holdout = split.holdout)
}

private fun calibrate(
    service: ClassificationService,
    probe: Record,
    holdout: List<LabeledFeatures>,
) {
    println("--- 1. Calibrating confidences ---")
    val before = service.classify(record = probe)
    println("Uncalibrated: ${before.label.value} at ${number(before.confidence)}")

    val fitted = service.fitCalibration(heldOut = holdout)
    val after = service.classify(record = probe)
    println("Fitted temperature: ${number(fitted.temperature)}")
    println("Calibrated:   ${after.label.value} at ${number(after.confidence)}")
    println()
    println("The label is unchanged — temperature scaling is rank-preserving, so it never")
    println("changes which class wins, only how confident the model claims to be. Naive Bayes")
    println("multiplies hundreds of independent-looking features, so its raw softmax saturates")
    println("near 0 and 1 regardless of how right it actually is.")
    println()

}

private fun abstain(service: ClassificationService) {
    println("--- 2. Abstaining instead of guessing ---")
    val ambiguous = Record(values = mapOf("purpose" to "transfer", "iban" to "DE9999"))
    listOf(0.0, ABSTAIN_THRESHOLD).forEach { threshold ->
        val decision = service.classifyOrNull(record = ambiguous, minConfidence = threshold)
        val rendered = decision?.let { p -> "${p.label.value} (${number(p.confidence)})" } ?: "abstained"
        println("  minConfidence=${number(threshold)} -> $rendered")
    }
    println()

}

private fun explain(service: ClassificationService, probe: Record) {
    println("--- 3. Why this label ---")
    val explanation = service.explain(record = probe, limit = CONTRIBUTION_LIMIT)!!
    println("Predicted ${explanation.label.value} at ${number(explanation.probability)}")
    println("base ${number(explanation.base)}  +  contributions  =  total ${number(explanation.total)}")
    println()
    println("Buckets only (safe to log):")
    println("-".repeat(DIVIDER_WIDTH))
    explanation.contributions.forEach { contribution ->
        println("  bucket %-10d %s".format(contribution.featureIndex, number(contribution.contribution)))
    }
    println()

    val withText = service.explain(
        record = probe,
        limit = CONTRIBUTION_LIMIT,
        mode = AttributionModeEnum.WITH_NGRAMS,
    )!!
    println("With n-grams (clear text — treat as source data):")
    println("-".repeat(DIVIDER_WIDTH))
    withText.contributions.forEach { contribution ->
        println("  %-22s %s".format("'${contribution.ngram ?: "?"}'", number(contribution.contribution)))
    }
    println()
    println("Contributions are mean-centered, so a feature equally likely under every label")
    println("scores zero rather than dominating the ranking. Note the 'iban' field never appears:")
    println("it is declared PII, so it never enters the feature text at all.")
    println()

}

private fun measure(
    service: ClassificationService,
    holdout: List<LabeledFeatures>,
) {
    println("--- 4. Measuring the model ---")
    val report = ModelEvaluator().evaluate(classifier = service.classifier, holdout = holdout)
    println("holdout accuracy ${number(report.accuracy)}  top-${report.topK} ${number(report.topKAccuracy)}")
    println("log loss ${number(report.logLoss)}  brier ${number(report.brierScore)}  " +
        "ECE ${number(report.expectedCalibrationError)}")
    println()
}
