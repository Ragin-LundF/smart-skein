package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class CrfSequenceLabelerTest {

    private val tokenizer = TypedTokenizer()
    private val key = Tag(value = "KEY")
    private val value = Tag(value = "VALUE")

    private fun tokens(text: String): List<Token> {
        return tokenizer.tokenize(text = text)
    }

    private fun trained(epochs: Int = 200): CrfSequenceLabeler {
        val labeler = CrfSequenceLabeler()
        // Regularity to learn: a WORD keyword is a KEY, the following ALPHANUMERIC/AMOUNT is a VALUE.
        val examples = listOf(
            tokens(text = "customer ab12") to listOf(key, value),
            tokens(text = "contract xy99") to listOf(key, value),
            tokens(text = "amount 12.50") to listOf(key, value),
        )
        repeat(times = epochs) {
            examples.forEach { (sequence, tags) -> labeler.learn(tokens = sequence, tags = tags) }
        }
        return labeler
    }

    @Test
    internal fun `labels a trained sequence correctly`() {
        val labeler = trained()
        assertEquals(expected = listOf(key, value), actual = labeler.label(tokens = tokens(text = "customer ab12")))
    }

    @Test
    internal fun `generalizes the type regularity to an unseen sequence`() {
        val labeler = trained()
        // "amount 99.00" was not trained verbatim, but WORD→KEY and AMOUNT→VALUE were learned.
        assertEquals(expected = listOf(key, value), actual = labeler.label(tokens = tokens(text = "amount 99.00")))
    }

    @Test
    internal fun `affix features generalize a suffix-driven tag to an unseen word`() {
        val labeler = CrfSequenceLabeler()
        val action = Tag(value = "ACTION")
        val noun = Tag(value = "NOUN")
        val examples = listOf(
            tokens(text = "running") to listOf(action),
            tokens(text = "jumping") to listOf(action),
            tokens(text = "walking") to listOf(action),
            tokens(text = "table") to listOf(noun),
            tokens(text = "chair") to listOf(noun),
            tokens(text = "house") to listOf(noun),
        )
        repeat(times = 200) { examples.forEach { (sequence, tags) -> labeler.learn(tokens = sequence, tags = tags) } }
        // "swimming" is unseen (its word/prefix features are untrained); only the "ing" suffix can drive the tag.
        assertEquals(expected = listOf(action), actual = labeler.label(tokens = tokens(text = "swimming")))
    }

    @Test
    internal fun `returns no tags for an empty token list`() {
        assertTrue(actual = trained(epochs = 1).label(tokens = emptyList()).isEmpty())
    }

    @Test
    internal fun `rejects labeling before training`() {
        assertFailsWith<IllegalStateException> {
            CrfSequenceLabeler().label(tokens = tokens(text = "anything"))
        }
    }

    @Test
    internal fun `rejects learning when tokens and tags differ in length`() {
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler().learn(tokens = tokens(text = "a b c"), tags = listOf(key))
        }
    }
}
