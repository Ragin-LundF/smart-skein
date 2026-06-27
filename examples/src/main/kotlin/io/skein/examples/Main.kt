package io.skein.examples

/** Runs the transaction classify → route → extract pipeline on a few sample transactions. */
fun main() {
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
