package io.skein.classify.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class SparseMatrixTest {

    private fun vector(vararg entries: Pair<Int, Float>): FeatureVector {
        return FeatureVector(
            indices = entries.map { (index, _) -> index }.toIntArray(),
            values = entries.map { (_, value) -> value }.toFloatArray(),
        )
    }

    private fun rowOf(matrix: SparseMatrix, row: Int): List<Pair<Int, Float>> {
        val block = matrix.blocks.first { candidate ->
            row >= candidate.firstRow && row < candidate.firstRow + candidate.rowCount()
        }
        val local = row - block.firstRow
        return (block.rowOffsets[local] until block.rowOffsets[local + 1])
            .map { k -> block.columnIndices[k] to block.values[k] }
    }

    @Test
    internal fun `round-trips the rows it was given`() {
        val matrix = SparseMatrixBuilder(columnCount = 10)
            .addRow(vector = vector(0 to 1.0f, 4 to 2.5f))
            .addRow(vector = vector(9 to -1.0f))
            .build()

        assertEquals(expected = 2, actual = matrix.rowCount)
        assertEquals(expected = 10, actual = matrix.columnCount)
        assertEquals(expected = 3L, actual = matrix.nonZeroCount())
        assertEquals(expected = listOf(0 to 1.0f, 4 to 2.5f), actual = rowOf(matrix = matrix, row = 0))
        assertEquals(expected = listOf(9 to -1.0f), actual = rowOf(matrix = matrix, row = 1))
    }

    @Test
    internal fun `keeps an empty row as an empty row rather than dropping it`() {
        val matrix = SparseMatrixBuilder(columnCount = 4)
            .addRow(vector = vector(1 to 1.0f))
            .addRow(vector = vector())
            .addRow(vector = vector(2 to 1.0f))
            .build()

        assertEquals(expected = 3, actual = matrix.rowCount)
        assertTrue(actual = rowOf(matrix = matrix, row = 1).isEmpty())
        assertEquals(expected = listOf(2 to 1.0f), actual = rowOf(matrix = matrix, row = 2))
    }

    @Test
    internal fun `splits into blocks once a block would exceed its non-zero limit`() {
        val builder = SparseMatrixBuilder(columnCount = 8, maxBlockNonZeros = 4)
        repeat(times = 5) { row ->
            builder.addRow(vector = vector(row to 1.0f, (row + 1) to 2.0f))
        }

        val matrix = builder.build()

        // Two non-zeros per row, so a four-non-zero block holds two rows: 2 + 2 + 1.
        assertEquals(expected = 3, actual = matrix.blocks.size)
        assertEquals(expected = listOf(0, 2, 4), actual = matrix.blocks.map { block -> block.firstRow })
        assertEquals(expected = 5, actual = matrix.rowCount)
        assertEquals(expected = 10L, actual = matrix.nonZeroCount())
        assertEquals(expected = listOf(4 to 1.0f, 5 to 2.0f), actual = rowOf(matrix = matrix, row = 4))
    }

    @Test
    internal fun `every block reports offsets consistent with its own row count`() {
        val builder = SparseMatrixBuilder(columnCount = 8, maxBlockNonZeros = 4)
        repeat(times = 5) { row -> builder.addRow(vector = vector(row to 1.0f, (row + 1) to 2.0f)) }

        val matrix = builder.build()

        matrix.blocks.forEach { block ->
            assertEquals(expected = block.rowCount() + 1, actual = block.rowOffsets.size)
            assertEquals(expected = block.columnIndices.size, actual = block.rowOffsets.last())
        }
    }

    @Test
    internal fun `selectRows cuts the given rows in the given order`() {
        val matrix = SparseMatrixBuilder(columnCount = 5)
            .addRow(vector = vector(0 to 1.0f))
            .addRow(vector = vector(1 to 2.0f))
            .addRow(vector = vector(2 to 3.0f))
            .build()

        val cut = matrix.selectRows(rows = intArrayOf(2, 0))

        assertEquals(expected = 2, actual = cut.rowCount)
        assertEquals(expected = 5, actual = cut.columnCount)
        assertEquals(expected = listOf(2 to 3.0f), actual = rowOf(matrix = cut, row = 0))
        assertEquals(expected = listOf(0 to 1.0f), actual = rowOf(matrix = cut, row = 1))
    }

    @Test
    internal fun `selectRows reaches across block boundaries`() {
        val builder = SparseMatrixBuilder(columnCount = 8, maxBlockNonZeros = 2)
        repeat(times = 6) { row -> builder.addRow(vector = vector(row to (row + 1).toFloat())) }

        val cut = builder.build().selectRows(rows = intArrayOf(0, 5, 3))

        assertEquals(expected = listOf(0 to 1.0f), actual = rowOf(matrix = cut, row = 0))
        assertEquals(expected = listOf(5 to 6.0f), actual = rowOf(matrix = cut, row = 1))
        assertEquals(expected = listOf(3 to 4.0f), actual = rowOf(matrix = cut, row = 2))
    }

    @Test
    internal fun `selectRows of nothing yields an empty matrix that still knows its width`() {
        val matrix = SparseMatrixBuilder(columnCount = 7).addRow(vector = vector(0 to 1.0f)).build()

        val cut = matrix.selectRows(rows = intArrayOf())

        assertEquals(expected = 0, actual = cut.rowCount)
        assertEquals(expected = 7, actual = cut.columnCount)
        assertEquals(expected = 0L, actual = cut.nonZeroCount())
    }

    @Test
    internal fun `rejects a row index outside the matrix`() {
        val matrix = SparseMatrixBuilder(columnCount = 3).addRow(vector = vector(0 to 1.0f)).build()

        assertFailsWith<IllegalArgumentException> { matrix.selectRows(rows = intArrayOf(1)) }
        assertFailsWith<IllegalArgumentException> { matrix.selectRows(rows = intArrayOf(-1)) }
    }

    /**
     * The one limit blocking cannot remove. The message must name it: a caller who hits this needs
     * to know it is a block-size ceiling, not an out-of-memory condition or an opaque
     * `NegativeArraySizeException` from an overflowed allocation.
     */
    @Test
    internal fun `rejects a row too wide for any block and names the limit`() {
        val builder = SparseMatrixBuilder(columnCount = 100, maxBlockNonZeros = 3)

        val failure = assertFailsWith<IllegalArgumentException> {
            builder.addRow(vector = vector(0 to 1.0f, 1 to 1.0f, 2 to 1.0f, 3 to 1.0f))
        }

        assertTrue(
            actual = failure.message!!.contains(other = "block limit"),
            message = "message did not name the block limit: ${failure.message}",
        )
    }

    @Test
    internal fun `grows past its initial capacity without losing rows`() {
        val builder = SparseMatrixBuilder(columnCount = 4096)
        repeat(times = 3000) { row -> builder.addRow(vector = vector(row to 1.0f)) }

        val matrix = builder.build()

        assertEquals(expected = 3000, actual = matrix.rowCount)
        assertEquals(expected = 3000L, actual = matrix.nonZeroCount())
        assertEquals(expected = listOf(2999 to 1.0f), actual = rowOf(matrix = matrix, row = 2999))
    }

    @Test
    internal fun `rejects an invalid shape`() {
        assertFailsWith<IllegalArgumentException> { SparseMatrixBuilder(columnCount = 0) }
        assertFailsWith<IllegalArgumentException> {
            SparseMatrixBuilder(columnCount = 4, maxBlockNonZeros = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            SparseMatrixBlock(
                firstRow = 0,
                rowOffsets = intArrayOf(0, 1),
                columnIndices = intArrayOf(0),
                values = floatArrayOf(1.0f, 2.0f),
            )
        }
        assertFailsWith<IllegalArgumentException> {
            SparseMatrixBlock(
                firstRow = 0,
                rowOffsets = intArrayOf(0, 5),
                columnIndices = intArrayOf(0),
                values = floatArrayOf(1.0f),
            )
        }
    }

    @Test
    internal fun `rejects an inverted row range`() {
        val builder = SparseMatrixBuilder(columnCount = 4)

        assertFailsWith<IllegalArgumentException> {
            builder.addRow(columnIndices = intArrayOf(0, 1), values = floatArrayOf(1.0f, 2.0f), from = 2, to = 0)
        }
    }
}
