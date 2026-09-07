package io.skein.cli

import io.skein.classify.application.ModelStore
import io.skein.classify.application.SchemaInference
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import java.nio.file.Path
import java.util.Locale
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.random.Random
import kotlin.system.exitProcess
import kotlin.time.measureTimedValue

private const val DEFAULT_BUDGET = 20
private const val DEFAULT_BATCH = 8
private const val DEFAULT_EPOCHS = 5
private const val DEFAULT_SCAN_LIMIT = 0
private const val CONFIDENCE_DECIMALS = 4
private const val DEFAULT_EVAL_TOP_K = 3
private const val DEFAULT_EVAL_BINS = 10
private const val DEFAULT_EVAL_TEST_RATIO = 0.2
private const val DEFAULT_EVAL_SEED = 42L
/** Process exit code: the command produced its result and any quality gate passed. */
internal const val EXIT_OK = 0
/** Process exit code: a report was produced but `--min-accuracy` was not met. */
internal const val EXIT_QUALITY_GATE_FAILED = 2

private val USAGE = """
    Skein CLI — train and inspect classifiers.

    Usage: skein-cli <command> [--flag value ...]

    Commands:
      label    Active-learning loop: confirm/correct the model's least-certain rows.
        --input <csv>        input records (required)
        --label-col <name>   the label column (required)
        --out <csv>          where to write the enriched, now-labeled records (required)
        --model <file>       model file: loaded if it exists (resume), saved at the end
        --classifier nb|logreg   classifier for a fresh model (default nb; ignored when resuming)
        --budget <n>         max rows to label this run (default $DEFAULT_BUDGET)
        --batch <n>          rows surfaced per re-selection (default $DEFAULT_BATCH)
        --strategy margin|least-confidence|entropy   uncertainty measure (default margin)
        --epochs <n>         SGD passes for logreg (default $DEFAULT_EPOCHS)
        --key <k0>,<k1>      fixed hashing key for a fresh model (default: random, then saved)
        --scan-limit <n>     cap rows scored per round; 0 = score the whole pool (default $DEFAULT_SCAN_LIMIT).
                             Set this (e.g. 100000) for multi-million-row pools to bound per-round work.
        --delimiter <char>   CSV field delimiter (default ,). Use \\t for tab-separated files.

      predict  Classify every input row using a saved model.
        --input <csv>        input records (required)
        --model <file>       saved model (required)
        --out <csv>          where to write rows with predicted label + confidence (required)
        --epochs <n>         SGD passes when rebuilding a logreg model (default $DEFAULT_EPOCHS)
        --min-confidence <p> abstain below this calibrated confidence (0..1, default 0 = label every
                             row). An abstained row gets an empty label cell; its confidence is still
                             written, and `label` will pick it up as pending.
        --delimiter <char>   CSV field delimiter (default ,). Use \\t for tab-separated files.

      export   Write a .skein model as human-readable text for inspection.
        --model <file>       source .skein model (required)
        --out <file>         destination text file (required)

      evaluate Measure model quality and print a report.
        --model <file>       saved .skein model (required)
        --input <csv>        labeled CSV to score the SAVED model against, as it is.
                             Mutually exclusive with --folds / --test-ratio.
        --folds <n>          stratified k-fold cross-validation over the model's stored
                             observations, retraining a fresh model per fold (n >= 2)
        --test-ratio <r>     stratified holdout over the stored observations (default $DEFAULT_EVAL_TEST_RATIO)
        --top-k <n>          rank depth for top-k accuracy (default $DEFAULT_EVAL_TOP_K)
        --bins <n>           calibration bins (default $DEFAULT_EVAL_BINS)
        --seed <n>           split seed (default $DEFAULT_EVAL_SEED)
        --epochs <n>         training passes over a split or fold (default $DEFAULT_EPOCHS)
        --out <file>         also write the text report here
        --csv <file>         write per-label metrics as CSV
        --confusion <file>   write the full confusion matrix as CSV
        --min-accuracy <r>   exit with code 2 when accuracy falls below r (CI quality gate)
        --delimiter <char>   CSV field delimiter (default ,). Use \\t for tab-separated files.

      Note: a .skein file stores the TRAINING observations. --folds and --test-ratio therefore
      measure the recipe (classifier + corpus + settings) by retraining, not the saved model
      itself, which already saw every stored row. Use --input to score the saved model.
""".trimIndent()

// ponytail: hand-rolled `when` dispatch + flag parsing (unknown flags are rejected, not ignored).
// Four commands now. Promote to a CLI library (clikt, picocli) if grouped sub-commands or
// positional arguments appear; plain `--flag value` parsing with unknown-flag rejection still
// covers everything here.
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println(USAGE)
        return
    }
    runCatching {
        val flags = parseFlags(tokens = args.drop(n = 1))
        val outcome = measureTimedValue { dispatch(command = args.first(), flags = flags) }
        println("✓ Completed in ${outcome.duration.absoluteValue}")
        outcome.value
    }.onFailure { error ->
        System.err.println("error: ${error.message}\n\n${error.stackTraceToString()}")
        exitProcess(status = 1)
    }.onSuccess { status ->
        if (status != EXIT_OK) {
            exitProcess(status = status)
        }
    }
}

/** Runs one command and returns its exit code. The only `exitProcess` call site is [main]. */
private fun dispatch(command: String, flags: Map<String, String>): Int {
    return when (command) {
        "label" -> {
            runLabel(flags = flags)
            EXIT_OK
        }

        "predict" -> {
            runPredict(flags = flags)
            EXIT_OK
        }

        "export" -> {
            runExport(flags = flags)
            EXIT_OK
        }

        "evaluate" -> runEvaluate(flags = flags)

        else -> {
            println(USAGE)
            EXIT_OK
        }
    }
}

private fun runLabel(flags: Map<String, String>) {
    requireKnownFlags(flags = flags, allowed = LABEL_FLAGS)
    val inputPath = Path(flags.required(name = "input"))
    val labelColumn = flags.required(name = "label-col")
    val outPath = Path(flags.required(name = "out"))
    val modelPath = flags["model"]?.let { value -> Path(value) }
    val epochs = flags["epochs"]?.toInt() ?: DEFAULT_EPOCHS

    val delimiter = parseDelimiter(value = flags["delimiter"])
    val source = CsvRecordSource(text = inputPath.readText(), delimiter = delimiter)
    require(value = source.header.isNotEmpty()) { "input CSV has no header row" }
    if (labelColumn !in source.header) {
        System.err.println(
            "warning: --label-col '$labelColumn' is not in the input header ${source.header}; " +
                    "treating all rows as unlabeled",
        )
    }

    val engine = if (modelPath != null && modelPath.exists()) {
        CliEngine.restore(model = ModelStore.load(path = modelPath), epochs = epochs)
    } else {
        val records = source.rows.map { row -> Record(values = row) }
        val schema = SchemaInference().infer(records = records, labelField = labelColumn)
        CliEngine.fresh(
            schema = schema,
            classifier = parseClassifier(value = flags["classifier"]),
            hashingConfig = parseKey(value = flags["key"]) ?: HashingConfig.randomKey(),
        )
    }

    val labeled = LabelingSession(
        engine = engine,
        budget = flags["budget"]?.toInt() ?: DEFAULT_BUDGET,
        batchSize = flags["batch"]?.toInt() ?: DEFAULT_BATCH,
        strategy = parseStrategy(value = flags["strategy"]),
        epochs = epochs,
        scanLimit = flags["scan-limit"]?.toInt() ?: DEFAULT_SCAN_LIMIT,
        random = Random.Default,
        input = System.`in`.bufferedReader(),
        output = System.out,
    ).run(rows = source.rows)

    val header = withColumn(header = source.header, column = engine.labelColumn)
    writeRows(outPath = outPath, header = header, rows = source.rows, delimiter = delimiter)
    modelPath?.let { path -> engine.save(path = path) }
    reportLabelOutcome(
        engine = engine,
        labeled = labeled,
        outPath = outPath.toString(),
        modelPath = modelPath?.toString(),
    )
}

private fun runPredict(flags: Map<String, String>) {
    requireKnownFlags(flags = flags, allowed = PREDICT_FLAGS)
    val inputPath = Path(flags.required(name = "input"))
    val outPath = Path(flags.required(name = "out"))
    val engine = CliEngine.restore(
        model = ModelStore.load(path = Path(flags.required(name = "model"))),
        epochs = flags["epochs"]?.toInt() ?: DEFAULT_EPOCHS,
    )
    val labelColumn = engine.labelColumn
    val confidenceColumn = "${labelColumn}_confidence"
    val delimiter = parseDelimiter(value = flags["delimiter"])
    val source = CsvRecordSource(text = inputPath.readText(), delimiter = delimiter)
    val minConfidence = parseAccuracyGate(value = flags["min-confidence"]) ?: 0.0
    val abstained = predictRows(
        engine = engine,
        rows = source.rows,
        labelColumn = labelColumn,
        confidenceColumn = confidenceColumn,
        minConfidence = minConfidence,
    )
    val header = withColumn(
        header = withColumn(header = source.header, column = labelColumn),
        column = confidenceColumn,
    )
    writeRows(outPath = outPath, header = header, rows = source.rows, delimiter = delimiter)
    val note = if (abstained == 0) "" else " ($abstained abstained below --min-confidence $minConfidence)"
    println("Wrote ${source.rows.size} predictions to $outPath$note")
}

/**
 * Fills the label and confidence columns of [rows] in place and returns how many rows abstained.
 *
 * A row whose confidence falls below [minConfidence] gets an **empty** label cell while still
 * carrying its confidence, so the operator can see how close it came. Blank is deliberate: the
 * output feeds straight back into `skein-cli label`, which treats an empty label as pending — so
 * `predict --min-confidence` followed by `label` hand-labels exactly the rows the model refused.
 */
internal fun predictRows(
    engine: CliEngine,
    rows: List<MutableMap<String, Any?>>,
    labelColumn: String,
    confidenceColumn: String,
    minConfidence: Double,
): Int {
    var abstained = 0
    rows.forEach { row ->
        val prediction = engine.service.classify(record = Record(values = row))
        val confident = prediction.isConfident(minConfidence = minConfidence)
        if (!confident) {
            abstained += 1
        }
        row[labelColumn] = if (confident) prediction.label.value else ""
        row[confidenceColumn] = "%.${CONFIDENCE_DECIMALS}f".format(Locale.ROOT, prediction.confidence)
    }
    return abstained
}

private fun runExport(flags: Map<String, String>) {
    requireKnownFlags(flags = flags, allowed = EXPORT_FLAGS)
    val modelPath = Path(flags.required(name = "model"))
    val outPath = Path(flags.required(name = "out"))
    ModelConverter.toText(src = modelPath, dst = outPath)
    println("Exported model to $outPath")
}

private fun reportLabelOutcome(engine: CliEngine, labeled: Int, outPath: String, modelPath: String?) {
    val metrics = engine.service.metrics()
    println("Labeled $labeled row(s). Wrote dataset to $outPath.")
    println("Model now holds ${metrics.totalObservations} observation(s):")
    metrics.perLabelCounts.toSortedMap(comparator = compareBy { label -> label.value })
        .forEach { (label, count) -> println("  ${label.value}: $count") }
    if (modelPath == null) {
        println("Note: no --model given, so the trained model was not saved.")
    } else {
        println("Saved model to $modelPath.")
    }
}

private fun writeRows(
    outPath: Path,
    header: List<String>,
    rows: List<Map<String, Any?>>,
    delimiter: Char = ','
) {
    outPath.writeText(text = CsvWriter(header = header, delimiter = delimiter).write(rows = rows))
}

private fun withColumn(header: List<String>, column: String): List<String> {
    return if (column in header) header else header + column
}
