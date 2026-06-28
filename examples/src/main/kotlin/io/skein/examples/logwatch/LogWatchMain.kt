package io.skein.examples.logwatch

import java.util.Locale
import kotlin.io.path.Path
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.io.path.writeText

private const val DEFAULT_MODEL = "log-anomaly.skein"
private const val DEFAULT_MIN_CONFIDENCE = 0.5
private const val LABEL_COLUMN_WIDTH = 15
private const val CONFIDENCE_FLAG = "--confidence"
private const val LOG_FLAG = "--log"
private const val RULES_FLAG = "--rules"
private const val MODEL_FLAG = "--model"
private const val OUT_FLAG = "--out"

private val USAGE = """
    Log anomaly detector — train a classifier from keyword rules, then scan log files.

    Commands:
      train  [--rules  rules.csv]   CSV with columns: anomaly_type,level,keyword
             [--model  model.skein] where to save the trained model (default: $DEFAULT_MODEL)

      scan   --log     app.log      log file to scan (or --sample to use the bundled example)
             [--model  model.skein] trained model to load (default: $DEFAULT_MODEL)
             [--out    results.csv] write flagged lines to CSV instead of stdout
             [--confidence 0.5]    minimum confidence to flag a line (default: $DEFAULT_MIN_CONFIDENCE)

    Usage:
      examples logwatch train
      examples logwatch train --rules my-rules.csv
      examples logwatch scan --sample
      examples logwatch scan --log /var/log/app.log --out anomalies.csv
""".trimIndent()

fun logWatchMain(args: Array<String>) {
    when (args.firstOrNull()) {
        "train" -> runTrain(args = args)
        "scan" -> runScan(args = args)
        else -> println(USAGE)
    }
}

private fun runTrain(args: Array<String>) {
    val rulesText = flagValue(args, RULES_FLAG)
        ?.let { path -> Path(path).readText() }
        ?: resource("logwatch/rules.csv")
    val modelPath = Path(flagValue(args, MODEL_FLAG) ?: DEFAULT_MODEL)

    print("Training on rules ... ")
    val (service, store) = trainFromRulesCsv(csv = rulesText)
    val labels = service.metrics().perLabelCounts.size
    println("done. $labels anomaly types learned.")

    saveModel(store = store, path = modelPath)
    println("Model saved → $modelPath")
    println("Run: examples logwatch scan --log <app.log>")
}

private fun runScan(args: Array<String>) {
    val modelPath = Path(flagValue(args, MODEL_FLAG) ?: DEFAULT_MODEL)
    require(modelPath.exists()) { "model not found at $modelPath — run 'logwatch train' first" }
    val minConfidence = flagValue(args, CONFIDENCE_FLAG)?.toDouble() ?: DEFAULT_MIN_CONFIDENCE
    val outFlag = flagValue(args, OUT_FLAG)

    val logText = when {
        "--sample" in args -> resource("logwatch/sample.log")
        flagValue(args, LOG_FLAG) != null -> Path(flagValue(args, LOG_FLAG)!!).readText()
        else -> error("provide --log <file> or --sample")
    }

    print("Loading model from $modelPath ... ")
    val service = loadModel(path = modelPath)
    println("done.")

    val lines = logText.lines().filter { it.isNotBlank() }
    val predictions = lines.mapNotNull { line -> service.classifyLine(line) }
    val flagged = predictions.filter { it.label != NORMAL_LABEL && it.confidence >= minConfidence }

    println("\nScanned ${lines.size} lines → ${flagged.size} anomalies (confidence ≥ $minConfidence):\n")

    if (outFlag != null) {
        writeCsv(flagged = flagged, path = outFlag)
        println("Results written to $outFlag")
    } else {
        flagged.forEach { p ->
            val pct = "%.0f".format(Locale.ROOT, p.confidence * 100)
            println("  [${p.label.padEnd(LABEL_COLUMN_WIDTH)}] $pct%  ${p.line}")
        }
    }
}

private fun writeCsv(flagged: List<LogPrediction>, path: String) {
    val header = "level,anomaly_type,confidence,log_line"
    val rows = flagged.map { p ->
        listOf(p.level, p.label, "%.2f".format(Locale.ROOT, p.confidence), p.line)
            .joinToString(",") { v -> if (',' in v || '"' in v) "\"${v.replace("\"", "\"\"")}\"" else v }
    }
    Path(path).writeText((listOf(header) + rows).joinToString("\n"))
}

private fun flagValue(args: Array<String>, name: String): String? {
    val i = args.indexOf(name)
    return if (i >= 0) args.getOrNull(i + 1) else null
}

private fun resource(name: String): String =
    LogWatchMain::class.java.classLoader.getResourceAsStream(name)
        ?.bufferedReader()?.readText()
        ?: error("bundled resource not found: $name")

private object LogWatchMain
