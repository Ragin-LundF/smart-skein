package io.skein.text.application

import io.skein.text.domain.FrequencyModel
import kotlin.test.Test
import kotlin.test.assertEquals

internal class BrokenWordRepairerTest {

    private fun repairerKnowing(vararg words: String): BrokenWordRepairer {
        val model = FrequencyModel()
        model.learnAll(words = words.toList())
        return BrokenWordRepairer(frequencyModel = model)
    }

    @Test
    internal fun `merges a wrongly split word into the known word`() {
        val repairer = repairerKnowing("apartment")
        assertEquals(expected = "apartment", actual = repairer.repair(text = "apart ment"))
    }

    @Test
    internal fun `keeps genuinely separate known words apart`() {
        val repairer = repairerKnowing("the", "payment")
        assertEquals(expected = "the payment", actual = repairer.repair(text = "the payment"))
    }

    @Test
    internal fun `repairs a split word inside a sentence`() {
        val repairer = repairerKnowing("the", "apartment", "is", "ready")
        assertEquals(
            expected = "the apartment is ready",
            actual = repairer.repair(text = "the apart ment is ready"),
        )
    }

    @Test
    internal fun `merges three fragments into one compound`() {
        val repairer = repairerKnowing("understanding")
        assertEquals(expected = "understanding", actual = repairer.repair(text = "under stand ing"))
    }

    @Test
    internal fun `leaves an unknown standalone token unchanged`() {
        val repairer = repairerKnowing("the", "payment")
        assertEquals(expected = "xyz", actual = repairer.repair(text = "xyz"))
    }

    @Test
    internal fun `merges fragments that form a typo within edit distance`() {
        val repairer = repairerKnowing("apartment")
        // "apartmnt" is missing one letter (edit distance 1) yet should still be recognized as one word.
        assertEquals(expected = "apartmnt", actual = repairer.repair(text = "apart mnt"))
    }

    @Test
    internal fun `returns empty string for blank input`() {
        val repairer = repairerKnowing("the")
        assertEquals(expected = "", actual = repairer.repair(text = "   "))
    }
}
