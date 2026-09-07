package io.skein.classify.domain

/**
 * Counts of expected-versus-predicted label pairs.
 *
 * [labels] is the sorted union of every label that was expected *or* predicted, so a label the
 * model invents but never should have still occupies a row and a column. Sorting by
 * [Label.value] keeps text and CSV renderings stable and diffable across runs.
 *
 * ponytail: backed by a dense `n * n` array. n is the label count (tens in practice), so the
 * square is trivially small. Switch to a sparse map only if a use case ever reaches a few thousand
 * labels.
 */
class ConfusionMatrix private constructor(
    val labels: List<Label>,
    private val counts: IntArray,
) {

    private val indexByLabel: Map<Label, Int> = labels.withIndex().associate { (index, label) -> label to index }

    /** How often [expected] was the truth while the model said [predicted]. */
    fun count(expected: Label, predicted: Label): Int {
        val row = indexOf(label = expected)
        val column = indexOf(label = predicted)
        return counts[row * labels.size + column]
    }

    /** How often [label] was the ground truth — its support. */
    fun rowTotal(label: Label): Int {
        val row = indexOf(label = label)
        var total = 0
        for (column in labels.indices) {
            total += counts[row * labels.size + column]
        }
        return total
    }

    /** How often the model predicted [label], right or wrong. */
    fun columnTotal(label: Label): Int {
        val column = indexOf(label = label)
        var total = 0
        for (row in labels.indices) {
            total += counts[row * labels.size + column]
        }
        return total
    }

    /** How often [label] was both expected and predicted — the diagonal cell. */
    fun correct(label: Label): Int {
        return count(expected = label, predicted = label)
    }

    /** Total number of outcomes counted. */
    fun total(): Int {
        return counts.sum()
    }

    /**
     * The non-empty off-diagonal cells, largest first, capped at [limit] — the actionable slice of
     * a matrix that is otherwise too large to read. Each entry is `(expected, predicted, count)`.
     */
    fun topConfusions(limit: Int): List<Triple<Label, Label, Int>> {
        require(value = limit > 0) { "limit must be positive" }
        val confusions = ArrayList<Triple<Label, Label, Int>>()
        for (row in labels.indices) {
            for (column in labels.indices) {
                val cell = counts[row * labels.size + column]
                if (row != column && cell > 0) {
                    confusions.add(element = Triple(first = labels[row], second = labels[column], third = cell))
                }
            }
        }
        confusions.sortByDescending { confusion -> confusion.third }
        return confusions.take(n = limit)
    }

    private fun indexOf(label: Label): Int {
        return requireNotNull(value = indexByLabel[label]) { "unknown label ${label.value}" }
    }

    companion object {

        /** Tallies [outcomes] into a matrix over the union of their expected and predicted labels. */
        fun of(outcomes: List<PredictionOutcome>): ConfusionMatrix {
            val labels = outcomes
                .flatMap { outcome -> listOf(outcome.expected, outcome.prediction.label) }
                .distinct()
                .sortedBy { label -> label.value }
            val indexByLabel = labels.withIndex().associate { (index, label) -> label to index }
            val counts = IntArray(size = labels.size * labels.size)
            for (outcome in outcomes) {
                val row = indexByLabel.getValue(key = outcome.expected)
                val column = indexByLabel.getValue(key = outcome.prediction.label)
                counts[row * labels.size + column]++
            }
            return ConfusionMatrix(labels = labels, counts = counts)
        }
    }
}
