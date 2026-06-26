package io.skein.classify.domain

/**
 * Result of importing records from a source: the [accepted] mapped records ready for learning,
 * the [rejected] records with their reasons, and any non-fatal [warnings].
 */
data class ImportResult(
    val accepted: List<MappedRecord>,
    val rejected: List<RejectedRecord>,
    val warnings: List<String>,
) {

    fun acceptedCount(): Int {
        return accepted.size
    }

    fun rejectedCount(): Int {
        return rejected.size
    }
}
