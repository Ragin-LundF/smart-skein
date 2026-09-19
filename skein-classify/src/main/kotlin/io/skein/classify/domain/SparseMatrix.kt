package io.skein.classify.domain

/**
 * A sparse design matrix in compressed sparse row form, split into row [blocks].
 *
 * **Why blocks.** A single flat CSR pair caps at [Int.MAX_VALUE] non-zeros. At a measured ~250
 * active features per document that ceiling arrives at roughly 8.5 million documents, and the
 * failure mode is an opaque `NegativeArraySizeException` from an overflowed size computation rather
 * than anything that names the real limit. Splitting rows across blocks removes the ceiling for any
 * corpus that fits in memory at all, and retrofitting it later means touching every numerical loop
 * that reads the matrix — so it is built in from the start.
 *
 * Values are `float`, not `double`. The stored features do not need the precision — they are
 * counts, or counts scaled by a weighting — while every accumulator that reads them stays `double`.
 * That halves the largest structure in the training process at no measurable accuracy cost.
 *
 * Instances are immutable; build one with [SparseMatrixBuilder].
 */
class SparseMatrix internal constructor(
    val rowCount: Int,
    val columnCount: Int,
    val blocks: List<SparseMatrixBlock>,
) {

    init {
        require(value = rowCount >= 0) { "rowCount must not be negative" }
        require(value = columnCount > 0) { "columnCount must be positive, got $columnCount" }
        require(value = blocks.sumOf { block -> block.rowCount() } == rowCount) {
            "blocks must together hold exactly $rowCount rows"
        }
    }

    /** Total stored non-zeros. A [Long] because the whole point of blocking is to exceed an [Int]. */
    fun nonZeroCount(): Long {
        return blocks.sumOf { block -> block.columnIndices.size.toLong() }
    }

    /**
     * A matrix holding only [rows], in the order given — the operation that cuts a fold out of a
     * corpus.
     *
     * Copies rather than viewing. A view would avoid the copy but keep the whole corpus reachable
     * for as long as any fold is alive, which is the opposite of what a memory-bounded k-fold run
     * needs.
     */
    fun selectRows(rows: IntArray): SparseMatrix {
        val builder = SparseMatrixBuilder(columnCount = columnCount)
        rows.forEach { row ->
            require(value = row in 0 until rowCount) { "row $row is outside 0 until $rowCount" }
            val block = blockContaining(row = row)
            val local = row - block.firstRow
            builder.addRow(
                columnIndices = block.columnIndices,
                values = block.values,
                from = block.rowOffsets[local],
                to = block.rowOffsets[local + 1],
            )
        }
        return builder.build()
    }

    /**
     * The block holding [row].
     *
     * ponytail: a linear scan over blocks. Blocks are large — tens of millions of non-zeros each —
     * so even a billion-row corpus has a handful of them and a scan is faster than a binary search's
     * branch misprediction. Ceiling: this degrades if the block size is ever configured small enough
     * to produce thousands of blocks. Upgrade path: binary search over a `firstRow` array.
     */
    private fun blockContaining(row: Int): SparseMatrixBlock {
        return blocks.first { block -> row >= block.firstRow && row < block.firstRow + block.rowCount() }
    }
}
