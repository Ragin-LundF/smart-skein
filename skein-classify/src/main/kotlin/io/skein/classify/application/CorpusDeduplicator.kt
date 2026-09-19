package io.skein.classify.application

import io.skein.classify.domain.MultiLabeledText

/**
 * Collapses rows that are identical once masked, keeping one representative of each.
 *
 * **Why the ratio is so large.** Measured on the reference probe, 400,000 documents collapsed to
 * **21,420** — nineteen to one. Once reference numbers and dates are masked away
 * (see [TokenMasker]), an enormous share of a real export is literally the same document: the same
 * merchant, the same wording, the same labels, a different id. Training on all 400,000 copies
 * teaches nothing that the first copy did not and costs nineteen times the time.
 *
 * *(That particular 19x is inflated by the probe drawing from only 7,319 base documents. Expect a
 * smaller but still large factor on real data, and measure it — [DeduplicationResult.ratio] reports
 * it — rather than assuming this one transfers.)*
 *
 * Rows are keyed on the **masked text together with the label set**, never on text alone. Two rows
 * with identical wording and different labels are a genuine disagreement in the data — a
 * mislabelling, or a distinction the features cannot see — and silently discarding one of them
 * would hide it.
 *
 * Dropping duplicates outright also removes a bias towards high-volume merchants, which may or may
 * not be what you want: if the frequency of a pattern should influence the fit, keep
 * [DeduplicationResult.occurrences] and weight by it rather than discarding.
 */
class CorpusDeduplicator(private val masker: TokenMasker = TokenMasker()) {

    /**
     * Masks every row of [corpus] and returns one representative per distinct (text, labels) pair,
     * in first-seen order so the result is deterministic.
     */
    fun deduplicate(corpus: List<MultiLabeledText>): DeduplicationResult {
        require(value = corpus.isNotEmpty()) { "cannot deduplicate an empty corpus" }
        val representatives = LinkedHashMap<Key, MultiLabeledText>()
        val occurrences = LinkedHashMap<Key, Int>()
        corpus.forEach { row ->
            val masked = row.copy(featureText = masker.mask(text = row.featureText))
            val key = Key(featureText = masked.featureText, labels = masked.labels.map { it.value }.toSortedSet())
            representatives.putIfAbsent(key, masked)
            occurrences[key] = (occurrences[key] ?: 0) + 1
        }
        return DeduplicationResult(
            rows = representatives.values.toList(),
            occurrences = representatives.keys.map { key -> occurrences.getValue(key = key) },
            originalSize = corpus.size,
        )
    }

    /** Masked text plus its label set — a data class so equality and hashing come for free. */
    private data class Key(val featureText: String, val labels: Set<String>)
}
