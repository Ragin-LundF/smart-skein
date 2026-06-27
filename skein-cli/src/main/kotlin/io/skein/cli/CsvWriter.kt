package io.skein.cli

/**
 * Writes records back to CSV text using [header] for column order. Any column missing from a row is
 * emitted as an empty field, so a freshly labeled column appears for every row whether or not the
 * original input carried it.
 */
class CsvWriter(private val header: List<String>) {

    fun write(rows: List<Map<String, Any?>>): String {
        val lines = ArrayList<String>(rows.size + 1)
        lines.add(CsvCodec.formatRow(row = header))
        rows.forEach { row ->
            lines.add(CsvCodec.formatRow(row = header.map { name -> row[name]?.toString() ?: "" }))
        }
        return lines.joinToString(separator = "\n", postfix = "\n")
    }
}
