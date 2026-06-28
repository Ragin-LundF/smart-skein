package io.skein.examples

import io.skein.examples.logwatch.logWatchMain

private val USAGE = """
    Skein examples — runnable demonstrations of the classify/extract pipeline.

    Usage: examples <use-case> [args...]

    Use cases:
      transaction   Classify bank-transaction text → route → extract structured fields.
                    Training data is inline; no files needed. Covers the classify → route → extract pipeline.

      logwatch      Train an anomaly detector from a keyword rules CSV, then scan log files.
                    train [--rules rules.csv] [--model model.skein]
                    scan  --log app.log | --sample  [--out results.csv] [--confidence 0.5]
""".trimIndent()

fun main(args: Array<String>) {
    when (args.firstOrNull()) {
        "transaction" -> runTransactionExample()
        "logwatch" -> logWatchMain(args = args.drop(1).toTypedArray())
        else -> println(USAGE)
    }
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
