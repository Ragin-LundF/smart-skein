package io.skein.classify.domain

/** Non-zeros a single block may hold before the builder starts a new one. */
private const val DEFAULT_MAX_BLOCK_NON_ZEROS = 1 shl 24

/** Initial capacity of the growable arrays, large enough that a small corpus never regrows. */
private const val INITIAL_CAPACITY = 1024

/**
 * Accumulates rows into a [SparseMatrix], starting a new block before any array can approach
 * [Int.MAX_VALUE].
 *
 * Three deliberate choices, each measured on the reference corpus:
 * - Row offsets accumulate into an [IntArray] rather than a `List<Int>`. Boxing one `Integer` per
 *   row costs ~20 MB of garbage on a million-row corpus for nothing.
 * - Growth doubles, which briefly holds the old array alongside the new. [maxBlockNonZeros] bounds
 *   how large that transient can get; the default caps a block at ~134 MB of column indices and
 *   values together, so a regrow costs a few hundred megabytes rather than gigabytes.
 * - Values are stored as `float`, halving the matrix against a `double` payload.
 *
 * Not thread-safe: build one matrix per thread, or build on one thread and share the result, which
 * is immutable.
 */
class SparseMatrixBuilder(
    private val columnCount: Int,
    private val maxBlockNonZeros: Int = DEFAULT_MAX_BLOCK_NON_ZEROS,
) {

    init {
        require(value = columnCount > 0) { "columnCount must be positive, got $columnCount" }
        require(value = maxBlockNonZeros > 0) { "maxBlockNonZeros must be positive, got $maxBlockNonZeros" }
    }

    private val blocks = ArrayList<SparseMatrixBlock>()
    private var rowCount = 0

    private var blockFirstRow = 0
    private var blockRowCount = 0
    private var offsets = IntArray(size = INITIAL_CAPACITY)
    private var columns = IntArray(size = INITIAL_CAPACITY)
    private var values = FloatArray(size = INITIAL_CAPACITY)
    private var nonZeros = 0

    /** Appends [vector] as the next row. Entries are stored as given, so they stay sorted. */
    fun addRow(vector: FeatureVector): SparseMatrixBuilder {
        return addRow(
            columnIndices = vector.indices,
            values = vector.values,
            from = 0,
            to = vector.indices.size,
        )
    }

    /** Appends `columnIndices[from until to]` as the next row, avoiding an intermediate copy. */
    fun addRow(columnIndices: IntArray, values: FloatArray, from: Int, to: Int): SparseMatrixBuilder {
        val length = to - from
        require(value = length >= 0) { "row range $from until $to is inverted" }
        requireRoomForAnotherRow()
        requireFitsInABlock(length = length)
        if (nonZeros.toLong() + length > maxBlockNonZeros) {
            sealBlock()
        }
        ensureCapacity(additional = length)
        System.arraycopy(columnIndices, from, this.columns, nonZeros, length)
        System.arraycopy(values, from, this.values, nonZeros, length)
        nonZeros += length
        blockRowCount++
        rowCount++
        offsets[blockRowCount] = nonZeros
        return this
    }

    /** Freezes the accumulated rows into an immutable matrix. The builder must not be reused. */
    fun build(): SparseMatrix {
        if (blockRowCount > 0) {
            sealBlock()
        }
        return SparseMatrix(rowCount = rowCount, columnCount = columnCount, blocks = blocks.toList())
    }

    /**
     * Blocking removes the ceiling on total non-zeros, but the row *index* is still an `Int`.
     *
     * Name that limit rather than letting the counter wrap silently into negative row indices, which
     * would surface far downstream as a `NegativeArraySizeException` or an out-of-bounds read with
     * nothing to connect it to the corpus size that caused it.
     */
    private fun requireRoomForAnotherRow() {
        check(value = rowCount < Int.MAX_VALUE) {
            "a design matrix cannot hold more than ${Int.MAX_VALUE} rows; " +
                "deduplicate the corpus or train on a sample"
        }
    }

    /**
     * A row wider than a whole block cannot be represented, and that is the one hard limit blocking
     * does not remove. Name it, rather than letting an array allocation fail with a size the caller
     * cannot connect to anything.
     */
    private fun requireFitsInABlock(length: Int) {
        require(value = length <= maxBlockNonZeros) {
            "a single row holds $length non-zeros, above the $maxBlockNonZeros block limit; " +
                "raise maxBlockNonZeros or reduce the feature count of this record"
        }
    }

    private fun sealBlock() {
        blocks.add(
            element = SparseMatrixBlock(
                firstRow = blockFirstRow,
                rowOffsets = offsets.copyOf(newSize = blockRowCount + 1),
                columnIndices = columns.copyOf(newSize = nonZeros),
                values = values.copyOf(newSize = nonZeros),
            ),
        )
        blockFirstRow = rowCount
        blockRowCount = 0
        nonZeros = 0
        offsets = IntArray(size = INITIAL_CAPACITY)
        columns = IntArray(size = INITIAL_CAPACITY)
        values = FloatArray(size = INITIAL_CAPACITY)
    }

    private fun ensureCapacity(additional: Int) {
        if (offsets.size < blockRowCount + 2) {
            offsets = offsets.copyOf(newSize = offsets.size * 2)
        }
        if (columns.size >= nonZeros + additional) {
            return
        }
        var target = columns.size.toLong()
        while (target < nonZeros + additional) {
            target *= 2
        }
        val capped = minOf(a = target, b = maxBlockNonZeros.toLong()).toInt()
        columns = columns.copyOf(newSize = capped)
        values = values.copyOf(newSize = capped)
    }
}
