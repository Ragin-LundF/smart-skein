package io.skein.cli

import io.skein.classify.application.ClassifierFactory
import io.skein.classify.application.EvaluationReportFormatter
import io.skein.classify.application.LoadedModel
import io.skein.classify.application.ModelEvaluator
import io.skein.classify.application.ModelStore
import io.skein.classify.application.StratifiedSplitter
import io.skein.classify.domain.EvaluationReport
import io.skein.classify.domain.Record
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.Path
import kotlin.io.path.readText
import kotlin.io.path.writeText

private const val DEFAULT_EVAL_TOP_K = 3
private const val DEFAULT_EVAL_BINS = 10
private const val DEFAULT_EVAL_TEST_RATIO = 0.2
private const val DEFAULT_EVAL_SEED = 42L
private const val DEFAULT_EVAL_FOLDS = 5
private const val DEFAULT_EVAL_EPOCHS = 5
private const val METRIC_DECIMALS = 6

/**
 * Measures model quality and prints a report. Returns the process exit code so the quality gate is
 * testable without forking a JVM: [EXIT_OK], or [EXIT_QUALITY_GATE_FAILED] when `--min-accuracy`
 * is not met.
 */
internal fun runEvaluate(flags: Map<String, String>): Int {
    requireKnownFlags(flags = flags, allowed = EVALUATE_FLAGS)
    requireOneMode(flags = flags)

    val model = ModelStore.load(path = Path(flags.required(name = "model")))
    val evaluator = ModelEvaluator(
        topK = parsePositiveInt(name = "top-k", value = flags["top-k"], default = DEFAULT_EVAL_TOP_K),
        calibrationBins = parsePositiveInt(name = "bins", value = flags["bins"], default = DEFAULT_EVAL_BINS),
        splitter = StratifiedSplitter(seed = flags["seed"]?.toLong() ?: DEFAULT_EVAL_SEED),
    )
    val epochs = parsePositiveInt(name = "epochs", value = flags["epochs"], default = DEFAULT_EVAL_EPOCHS)
    val rendered = evaluateForMode(flags = flags, model = model, evaluator = evaluator, epochs = epochs)

    println(rendered.text)
    writeOutputs(flags = flags, rendered = rendered)
    return gateStatus(flags = flags, report = rendered.report)
}

/**
 * `--input` scores the saved model as it stands; the other two modes retrain over the stored
 * corpus. Mixing them would silently report one while the caller asked for the other.
 */
private fun requireOneMode(flags: Map<String, String>) {
    val hasInput = flags.containsKey(key = "input")
    val hasFolds = flags.containsKey(key = "folds")
    val hasRatio = flags.containsKey(key = "test-ratio")
    require(value = !(hasInput && (hasFolds || hasRatio))) {
        "--input scores the saved model as it is; drop --folds/--test-ratio to use it"
    }
    require(value = !(hasFolds && hasRatio)) { "choose one of --folds or --test-ratio" }
}

private fun evaluateForMode(
    flags: Map<String, String>,
    model: LoadedModel,
    evaluator: ModelEvaluator,
    epochs: Int,
): RenderedEvaluation {
    return when {
        flags.containsKey(key = "input") ->
            evaluateAgainstInput(flags = flags, model = model, evaluator = evaluator, epochs = epochs)

        flags.containsKey(key = "folds") ->
            crossValidateStored(flags = flags, model = model, evaluator = evaluator, epochs = epochs)

        else -> holdoutStored(flags = flags, model = model, evaluator = evaluator, epochs = epochs)
    }
}

private fun crossValidateStored(
    flags: Map<String, String>,
    model: LoadedModel,
    evaluator: ModelEvaluator,
    epochs: Int,
): RenderedEvaluation {
    val folds = parsePositiveInt(name = "folds", value = flags["folds"], default = DEFAULT_EVAL_FOLDS)
    val report = evaluator.crossValidate(
        observations = model.observations,
        classifierFactory = { ClassifierFactory.create(kind = model.classifier) },
        folds = folds,
        epochs = epochs,
    )
    return RenderedEvaluation(
        report = report.pooled,
        text = EvaluationReportFormatter.toText(
            report = report,
            title = "Evaluation — $folds-fold cross-validation over the stored corpus (retrained per fold)",
        ),
    )
}

private fun holdoutStored(
    flags: Map<String, String>,
    model: LoadedModel,
    evaluator: ModelEvaluator,
    epochs: Int,
): RenderedEvaluation {
    val ratio = parseRatio(name = "test-ratio", value = flags["test-ratio"], default = DEFAULT_EVAL_TEST_RATIO)
    val report = evaluator.holdout(
        observations = model.observations,
        classifierFactory = { ClassifierFactory.create(kind = model.classifier) },
        testRatio = ratio,
        epochs = epochs,
    )
    return RenderedEvaluation(
        report = report,
        text = EvaluationReportFormatter.toText(
            report = report,
            title = "Evaluation — stratified holdout $ratio over the stored corpus (retrained)",
        ),
    )
}

private fun writeOutputs(flags: Map<String, String>, rendered: RenderedEvaluation) {
    flags["out"]?.let { path -> Path(path).writeText(text = rendered.text) }
    flags["csv"]?.let { path -> writePerLabelCsv(report = rendered.report, outPath = Path(path)) }
    flags["confusion"]?.let { path -> writeConfusionCsv(report = rendered.report, outPath = Path(path)) }
}

private fun gateStatus(flags: Map<String, String>, report: EvaluationReport): Int {
    val gate = parseAccuracyGate(value = flags["min-accuracy"]) ?: return EXIT_OK
    if (report.accuracy >= gate) {
        return EXIT_OK
    }
    System.err.println(
        "quality gate failed: accuracy ${"%.4f".format(Locale.ROOT, report.accuracy)} " +
            "< --min-accuracy ${"%.4f".format(Locale.ROOT, gate)}",
    )
    return EXIT_QUALITY_GATE_FAILED
}

/** A report plus the text already rendered for it, so the caller renders once. */
private class RenderedEvaluation(val report: EvaluationReport, val text: String)

private fun evaluateAgainstInput(
    flags: Map<String, String>,
    model: LoadedModel,
    evaluator: ModelEvaluator,
    epochs: Int,
): RenderedEvaluation {
    val inputPath = Path(flags.required(name = "input"))
    val engine = CliEngine.restore(model = model, epochs = epochs)
    val delimiter = parseDelimiter(value = flags["delimiter"])
    val source = CsvRecordSource(text = inputPath.readText(), delimiter = delimiter)
    require(value = engine.labelColumn in source.header) {
        "input CSV has no '${engine.labelColumn}' column, so there is no ground truth to score against"
    }
    val records = source.rows.map { row -> Record(values = row) }
    val report = evaluator.evaluateRecords(service = engine.service, records = records)
    return RenderedEvaluation(
        report = report,
        text = EvaluationReportFormatter.toText(
            report = report,
            title = "Evaluation — saved model vs $inputPath (${records.size} rows)",
        ),
    )
}

private fun writePerLabelCsv(report: EvaluationReport, outPath: Path) {
    val header = listOf("label", "precision", "recall", "f1", "support")
    val rows = report.perLabel.map { metrics ->
        mapOf(
            "label" to metrics.label.value,
            "precision" to "%.${METRIC_DECIMALS}f".format(Locale.ROOT, metrics.precision),
            "recall" to "%.${METRIC_DECIMALS}f".format(Locale.ROOT, metrics.recall),
            "f1" to "%.${METRIC_DECIMALS}f".format(Locale.ROOT, metrics.f1),
            "support" to metrics.support.toString(),
        )
    }
    outPath.writeText(text = CsvWriter(header = header).write(rows = rows))
}

private fun writeConfusionCsv(report: EvaluationReport, outPath: Path) {
    val matrix = report.confusionMatrix
    val header = listOf("expected") + matrix.labels.map { label -> label.value }
    val rows = matrix.labels.map { expected ->
        val row = LinkedHashMap<String, Any?>()
        row["expected"] = expected.value
        matrix.labels.forEach { predicted ->
            row[predicted.value] = matrix.count(expected = expected, predicted = predicted).toString()
        }
        row
    }
    outPath.writeText(text = CsvWriter(header = header).write(rows = rows))
}
