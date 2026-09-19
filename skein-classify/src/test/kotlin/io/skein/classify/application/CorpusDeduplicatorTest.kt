package io.skein.classify.application

import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledText
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

internal class CorpusDeduplicatorTest {

    private val deduplicator = CorpusDeduplicator()
    private val rent = setOf(Label(value = "RENT"))

    private fun row(text: String, labels: Set<Label> = rent, group: String? = null): MultiLabeledText {
        return MultiLabeledText(featureText = text, labels = labels, group = group)
    }

    /**
     * The measured effect, in miniature: rows that differ only by a reference number are the same
     * row once masked, and training on all of them teaches nothing the first one did not.
     */
    @Test
    internal fun `collapses rows that differ only by a masked identifier`() {
        val corpus = listOf(
            row(text = "monthly rent ref 100200300"),
            row(text = "monthly rent ref 900800700"),
            row(text = "monthly rent ref 555444333"),
        )

        val result = deduplicator.deduplicate(corpus = corpus)

        assertEquals(expected = 1, actual = result.rows.size)
        assertEquals(expected = listOf(3), actual = result.occurrences)
        assertEquals(expected = 3.0, actual = result.ratio())
    }

    /**
     * Keying on text alone would silently discard one side of a genuine labelling disagreement,
     * which is exactly the thing a corpus audit needs to see.
     */
    @Test
    internal fun `keeps identical text carrying different labels`() {
        val corpus = listOf(
            row(text = "monthly payment", labels = rent),
            row(text = "monthly payment", labels = setOf(Label(value = "INSURANCE"))),
        )

        val result = deduplicator.deduplicate(corpus = corpus)

        assertEquals(expected = 2, actual = result.rows.size)
    }

    @Test
    internal fun `returns masked text so the survivors are what would be featurised`() {
        val result = deduplicator.deduplicate(corpus = listOf(row(text = "rent ref 100200300")))

        assertEquals(expected = "rent ref <NUMERIC>", actual = result.rows.single().featureText)
    }

    @Test
    internal fun `keeps the first occurrence so the result is deterministic`() {
        val corpus = listOf(
            row(text = "rent ref 111222333", group = "first"),
            row(text = "rent ref 444555666", group = "second"),
        )

        val result = deduplicator.deduplicate(corpus = corpus)

        assertEquals(expected = "first", actual = result.rows.single().group)
    }

    @Test
    internal fun `reports a ratio of one when nothing is duplicated`() {
        val corpus = listOf(row(text = "rent"), row(text = "insurance"), row(text = "groceries"))

        val result = deduplicator.deduplicate(corpus = corpus)

        assertEquals(expected = 3, actual = result.rows.size)
        assertEquals(expected = 1.0, actual = result.ratio())
        assertEquals(expected = listOf(1, 1, 1), actual = result.occurrences)
    }

    @Test
    internal fun `treats an unlabelled row as its own key`() {
        val corpus = listOf(row(text = "rent", labels = emptySet()), row(text = "rent", labels = rent))

        assertEquals(expected = 2, actual = deduplicator.deduplicate(corpus = corpus).rows.size)
    }

    @Test
    internal fun `rejects an empty corpus`() {
        assertFailsWith<IllegalArgumentException> { deduplicator.deduplicate(corpus = emptyList()) }
    }
}
