package io.skein.cli

import io.skein.classify.domain.Record
import io.skein.classify.spi.RecordSource

/**
 * Reads records from CSV text: the first row is the header (field names), every following row a
 * record. Each row is kept as a mutable map so a labeling session can fill in the label column in
 * place and write the enriched dataset back out.
 *
 * ponytail: parses the whole text up front (training sets a human labels by hand are small). Stream
 * row-by-row only if a corpus ever outgrows memory.
 */
class CsvRecordSource(text: String) : RecordSource {

    val header: List<String>
    val rows: List<MutableMap<String, Any?>>

    init {
        val parsed = CsvCodec.parse(text = text)
        if (parsed.isEmpty()) {
            header = emptyList()
            rows = emptyList()
        } else {
            header = parsed.first()
            rows = parsed.drop(n = 1).map { values -> toRow(values = values) }
        }
    }

    override fun stream(): Sequence<Record> {
        return rows.asSequence().map { row -> Record(values = row) }
    }

    private fun toRow(values: List<String>): MutableMap<String, Any?> {
        val row = LinkedHashMap<String, Any?>(header.size)
        header.forEachIndexed { index, name ->
            row[name] = values.getOrElse(index = index) { "" }
        }
        return row
    }
}
