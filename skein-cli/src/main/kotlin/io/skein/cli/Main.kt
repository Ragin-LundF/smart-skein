package io.skein.cli

import io.skein.classify.application.SchemaInference
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.UncertaintyStrategyEnum
import java.util.Locale
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText
import kotlin.system.exitProcess

private const val DEFAULT_BUDGET = 20
private const val DEFAULT_BATCH = 8
private const val DEFAULT_EPOCHS = 5
private const val CONFIDENCE_DECIMALS = 4

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

      predict  Classify every input row using a saved model.
        --input <csv>        input records (required)
        --model <file>       saved model (required)
        --out <csv>          where to write rows with predicted label + confidence (required)
        --epochs <n>         SGD passes when rebuilding a logreg model (default $DEFAULT_EPOCHS)
""".trimIndent()

// ponytail: hand-rolled `when` dispatch + flag parsing. Promote to a CLI library only if a third
// command or grouped sub-commands appear.
fun main(args: Array<String>) {
    if (args.isEmpty()) {
        println(USAGE)
        return
    }
    runCatching {
        val flags = parseFlags(tokens = args.drop(n = 1))
        when (args.first()) {
            "label" -> runLabel(flags = flags)
            "predict" -> runPredict(flags = flags)
            else -> println(USAGE)
        }
    }.onFailure { error ->
        System.err.println("error: ${error.message}")
        exitProcess(status = 1)
    }
}

private fun runLabel(flags: Map<String, String>) {
    val inputPath = Path(flags.required(name = "input"))
    val labelColumn = flags.required(name = "label-col")
    val outPath = Path(flags.required(name = "out"))
    val modelPath = flags["model"]?.let { value -> Path(value) }
    val epochs = flags["epochs"]?.toInt() ?: DEFAULT_EPOCHS

    val source = CsvRecordSource(text = inputPath.readText())
    require(value = source.header.isNotEmpty()) { "input CSV has no header row" }

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
        input = System.`in`.bufferedReader(),
        output = System.out,
    ).run(rows = source.rows)

    val header = withColumn(header = source.header, column = engine.labelColumn)
    writeRows(outPath = outPath, header = header, rows = source.rows)
    modelPath?.let { path -> engine.save(path = path) }
    reportLabelOutcome(
        engine = engine,
        labeled = labeled,
        outPath = outPath.toString(),
        modelPath = modelPath?.toString(),
    )
}

private fun runPredict(flags: Map<String, String>) {
    val inputPath = Path(flags.required(name = "input"))
    val outPath = Path(flags.required(name = "out"))
    val engine = CliEngine.restore(
        model = ModelStore.load(path = Path(flags.required(name = "model"))),
        epochs = flags["epochs"]?.toInt() ?: DEFAULT_EPOCHS,
    )
    val labelColumn = engine.labelColumn
    val confidenceColumn = "${labelColumn}_confidence"

    val source = CsvRecordSource(text = inputPath.readText())
    source.rows.forEach { row ->
        val prediction = engine.service.classify(record = Record(values = row))
        row[labelColumn] = prediction.label.value
        row[confidenceColumn] = "%.${CONFIDENCE_DECIMALS}f".format(Locale.ROOT, prediction.confidence)
    }
    val header = withColumn(
        header = withColumn(header = source.header, column = labelColumn),
        column = confidenceColumn,
    )
    writeRows(outPath = outPath, header = header, rows = source.rows)
    println("Wrote ${source.rows.size} predictions to $outPath")
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

private fun writeRows(outPath: java.nio.file.Path, header: List<String>, rows: List<Map<String, Any?>>) {
    outPath.writeText(text = CsvWriter(header = header).write(rows = rows))
}

private fun withColumn(header: List<String>, column: String): List<String> {
    return if (column in header) header else header + column
}

private fun parseFlags(tokens: List<String>): Map<String, String> {
    val flags = LinkedHashMap<String, String>()
    var index = 0
    while (index < tokens.size) {
        val token = tokens[index]
        require(value = token.startsWith(prefix = "--")) { "expected a --flag, got '$token'" }
        val name = token.removePrefix(prefix = "--")
        require(value = index + 1 < tokens.size) { "missing value for --$name" }
        flags[name] = tokens[index + 1]
        index += 2
    }
    return flags
}

private fun Map<String, String>.required(name: String): String {
    return this[name] ?: throw IllegalArgumentException("missing required --$name")
}

private fun parseClassifier(value: String?): ClassifierKindEnum {
    return when (value) {
        null, "nb" -> ClassifierKindEnum.NAIVE_BAYES
        "logreg" -> ClassifierKindEnum.LOGISTIC_REGRESSION
        else -> throw IllegalArgumentException("unknown --classifier '$value' (use nb or logreg)")
    }
}

private fun parseStrategy(value: String?): UncertaintyStrategyEnum {
    return when (value) {
        null, "margin" -> UncertaintyStrategyEnum.MARGIN
        "least-confidence" -> UncertaintyStrategyEnum.LEAST_CONFIDENCE
        "entropy" -> UncertaintyStrategyEnum.ENTROPY
        else -> throw IllegalArgumentException("unknown --strategy '$value'")
    }
}

private fun parseKey(value: String?): HashingConfig? {
    if (value == null) {
        return null
    }
    val parts = value.split(",")
    require(value = parts.size == 2) { "--key must be '<key0>,<key1>'" }
    return HashingConfig(key0 = parts[0].trim().toLong(), key1 = parts[1].trim().toLong())
}
