package io.skein.classify.domain

/**
 * One contiguous row range of a [SparseMatrix] in compressed sparse row form.
 *
 * The arrays are exposed directly, as [FeatureVector]'s are, because every consumer of this type is
 * a numerical inner loop where an accessor call per non-zero is the dominant cost.
 *
 * [rowOffsets] has one entry per row in this block plus a terminator, so row `r` of the block
 * occupies `columnIndices[rowOffsets[r] until rowOffsets[r + 1]]`. [firstRow] is that block's
 * offset into the owning matrix, so block-local row `r` is matrix row `firstRow + r`.
 */
class SparseMatrixBlock(
    val firstRow: Int,
    val rowOffsets: IntArray,
    val columnIndices: IntArray,
    val values: FloatArray,
) {

    init {
        require(value = rowOffsets.isNotEmpty()) { "rowOffsets must carry at least the terminator" }
        require(value = columnIndices.size == values.size) {
            "columnIndices and values must have equal length"
        }
        require(value = rowOffsets.last() == columnIndices.size) {
            "rowOffsets terminator ${rowOffsets.last()} must equal the non-zero count ${columnIndices.size}"
        }
    }

    /** Rows held by this block. */
    fun rowCount(): Int {
        return rowOffsets.size - 1
    }
}
