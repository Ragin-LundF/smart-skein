package io.skein.cli

/**
 * Minimal RFC-4180 CSV reader/writer: comma-separated, double-quoted fields, `""` escaping a quote
 * inside a quoted field, and quoted fields may span newlines. Sufficient for the tabular training
 * data this CLI labels.
 *
 * A leading UTF-8 BOM is stripped (Excel/Windows exports emit one), so the first header name is not
 * silently prefixed with U+FEFF.
 *
 * ponytail: minimal CSV (comma delimiter, UTF-8). Swap for a CSV library if inputs get exotic —
 * other delimiters, encodings, or embedded-NUL handling.
 */
object CsvCodec {

    private const val QUOTE = '"'
    private const val BOM = "\uFEFF"

    /** Parses [text] into rows of string fields. Empty input yields no rows. */
    fun parse(text: String, delimiter: Char = ','): List<List<String>> {
        val input = text.removePrefix(prefix = BOM)
        val scan = Scan(delimiter = delimiter)
        var index = 0
        while (index < input.length) {
            index += scan.step(text = input, index = index)
        }
        scan.endInput()
        return scan.rows
    }

    /** Formats one [row] of fields into a CSV line (no trailing newline). */
    fun formatRow(row: List<String>, delimiter: Char = ','): String {
        return row.joinToString(separator = delimiter.toString()) { value ->
            escape(
                value = value,
                delimiter = delimiter
            )
        }
    }

    private fun escape(value: String, delimiter: Char): String {
        val needsQuoting = value.any { character ->
            character == QUOTE || character == delimiter || character == '\n' || character == '\r'
        }
        if (!needsQuoting) {
            return value
        }
        return QUOTE + value.replace(oldValue = "\"", newValue = "\"\"") + QUOTE
    }

    /** Mutable cursor over the input; each [step] consumes one or two characters. */
    private class Scan(private val delimiter: Char) {

        val rows = ArrayList<List<String>>()
        private val field = StringBuilder()
        private var row = ArrayList<String>()
        private var inQuotes = false
        private var sawAnyChar = false

        /** Consumes the character at [index]; returns how many characters were consumed (1 or 2). */
        fun step(text: String, index: Int): Int {
            sawAnyChar = true
            val character = text[index]
            val next = peek(text = text, index = index + 1)
            return if (inQuotes) {
                stepQuoted(character = character, next = next)
            } else {
                stepUnquoted(character = character, next = next)
            }
        }

        /** Flushes any trailing field/row once the input ends (no trailing newline required). */
        fun endInput() {
            if (sawAnyChar || field.isNotEmpty() || row.isNotEmpty()) {
                endRow()
            }
        }

        private fun stepQuoted(character: Char, next: Char?): Int {
            if (character != QUOTE) {
                field.append(character)
                return 1
            }
            if (next == QUOTE) {
                field.append(QUOTE)
                return 2
            }
            inQuotes = false
            return 1
        }

        private fun stepUnquoted(character: Char, next: Char?): Int {
            return when (character) {
                QUOTE -> openQuote()
                delimiter -> endField()
                '\n' -> endRow()
                '\r' -> endRow() + if (next == '\n') 1 else 0
                else -> append(character = character)
            }
        }

        private fun openQuote(): Int {
            inQuotes = true
            return 1
        }

        private fun append(character: Char): Int {
            field.append(character)
            return 1
        }

        private fun endField(): Int {
            row.add(field.toString())
            field.clear()
            return 1
        }

        private fun endRow(): Int {
            endField()
            rows.add(row)
            row = ArrayList()
            sawAnyChar = false
            return 1
        }

        private fun peek(text: String, index: Int): Char? {
            return if (index < text.length) text[index] else null
        }
    }
}
