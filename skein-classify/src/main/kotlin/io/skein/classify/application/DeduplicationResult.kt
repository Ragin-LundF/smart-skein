package io.skein.classify.application

import io.skein.classify.domain.MultiLabeledText

/**
 * What [CorpusDeduplicator] found.
 *
 * [rows] holds the masked survivors and [occurrences] how many original rows each stood in for,
 * aligned by position — keep it if the frequency of a pattern should influence the fit, and ignore
 * it if it should not.
 *
 * Report [ratio] on real data rather than assuming the reference figure transfers. It is the number
 * that says how much of a corpus is genuinely distinct, and it decides whether an explicit
 * vocabulary is still viable at the corpus size in hand.
 */
data class DeduplicationResult(
    val rows: List<MultiLabeledText>,
    val occurrences: List<Int>,
    val originalSize: Int,
) {

    init {
        require(value = rows.size == occurrences.size) { "one occurrence count per surviving row is required" }
        require(value = originalSize > 0) { "originalSize must be positive" }
    }

    /** Original rows per survivor. `1.0` means the corpus held no duplicates at all. */
    fun ratio(): Double {
        return originalSize.toDouble() / rows.size
    }
}
