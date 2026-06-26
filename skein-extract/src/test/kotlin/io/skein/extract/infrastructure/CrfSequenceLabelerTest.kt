package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.Token
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

class CrfSequenceLabelerTest {

    private val tokenizer = TypedTokenizer()
    private val key = Tag("KEY")
    private val value = Tag("VALUE")

    private fun tokens(text: String): List<Token> {
        return tokenizer.tokenize(text)
    }

    private fun trained(epochs: Int = 200): CrfSequenceLabeler {
        val labeler = CrfSequenceLabeler()
        // Regularity to learn: a WORD keyword is a KEY, the following ALPHANUMERIC/AMOUNT is a VALUE.
        val examples = listOf(
            tokens("customer ab12") to listOf(key, value),
            tokens("contract xy99") to listOf(key, value),
            tokens("amount 12.50") to listOf(key, value),
        )
        repeat(epochs) {
            examples.forEach { (sequence, tags) -> labeler.learn(sequence, tags) }
        }
        return labeler
    }

    @Test
    fun `labels a trained sequence correctly`() {
        val labeler = trained()
        assertEquals(expected = listOf(key, value), actual = labeler.label(tokens("customer ab12")))
    }

    @Test
    fun `generalizes the type regularity to an unseen sequence`() {
        val labeler = trained()
        // "amount 99.00" was not trained verbatim, but WORD→KEY and AMOUNT→VALUE were learned.
        assertEquals(expected = listOf(key, value), actual = labeler.label(tokens("amount 99.00")))
    }

    @Test
    fun `affix features generalize a suffix-driven tag to an unseen word`() {
        val labeler = CrfSequenceLabeler()
        val action = Tag("ACTION")
        val noun = Tag("NOUN")
        val examples = listOf(
            tokens("running") to listOf(action),
            tokens("jumping") to listOf(action),
            tokens("walking") to listOf(action),
            tokens("table") to listOf(noun),
            tokens("chair") to listOf(noun),
            tokens("house") to listOf(noun),
        )
        repeat(200) { examples.forEach { (sequence, tags) -> labeler.learn(sequence, tags) } }
        // "swimming" is unseen (its word/prefix features are untrained); only the "ing" suffix can drive the tag.
        assertEquals(expected = listOf(action), actual = labeler.label(tokens("swimming")))
    }

    @Test
    fun `returns no tags for an empty token list`() {
        assertTrue(trained(epochs = 1).label(emptyList()).isEmpty())
    }

    @Test
    fun `rejects labeling before training`() {
        assertFailsWith<IllegalStateException> {
            CrfSequenceLabeler().label(tokens("anything"))
        }
    }

    @Test
    fun `rejects learning when tokens and tags differ in length`() {
        assertFailsWith<IllegalArgumentException> {
            CrfSequenceLabeler().learn(tokens("a b c"), listOf(key))
        }
    }
}
