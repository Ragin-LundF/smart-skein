package io.skein.cli

import kotlin.test.Test
import kotlin.test.assertEquals

internal class CsvCodecTest {

    @Test
    internal fun `parses quoted fields with embedded commas, quotes and newlines`() {
        val text = "a,b,c\n\"x,1\",\"say \"\"hi\"\"\",\"two\nlines\"\n"

        val rows = CsvCodec.parse(text = text)

        assertEquals(expected = listOf("a", "b", "c"), actual = rows[0])
        assertEquals(expected = listOf("x,1", "say \"hi\"", "two\nlines"), actual = rows[1])
    }

    @Test
    internal fun `format then parse round-trips fields needing escaping`() {
        val row = listOf("plain", "with,comma", "with\"quote", "with\nnewline")

        val parsed = CsvCodec.parse(text = CsvCodec.formatRow(row = row))

        assertEquals(expected = listOf(row), actual = parsed)
    }

    @Test
    internal fun `strips a leading UTF-8 BOM from the first header name`() {
        val rows = CsvCodec.parse(text = "\uFEFFa,b\n1,2\n")

        assertEquals(expected = listOf("a", "b"), actual = rows[0])
    }

    @Test
    internal fun `CsvRecordSource maps header columns to record fields`() {
        val source = CsvRecordSource(text = "purpose,category\nrent payment,rent\nsalary,\n")

        assertEquals(expected = listOf("purpose", "category"), actual = source.header)
        assertEquals(expected = 2, actual = source.rows.size)
        assertEquals(expected = "rent payment", actual = source.rows[0]["purpose"])
        assertEquals(expected = "rent", actual = source.rows[0]["category"])
        assertEquals(expected = "", actual = source.rows[1]["category"])
    }
}
