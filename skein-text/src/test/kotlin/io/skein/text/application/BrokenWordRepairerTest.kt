package io.skein.text.application

import io.skein.text.domain.FrequencyModel
import kotlin.test.Test
import kotlin.test.assertEquals

class BrokenWordRepairerTest {

    private fun repairerKnowing(vararg words: String): BrokenWordRepairer {
        val model = FrequencyModel()
        model.learnAll(words.toList())
        return BrokenWordRepairer(model)
    }

    @Test
    fun `merges a wrongly split word into the known word`() {
        val repairer = repairerKnowing("apartment")
        assertEquals(expected = "apartment", actual = repairer.repair("apart ment"))
    }

    @Test
    fun `keeps genuinely separate known words apart`() {
        val repairer = repairerKnowing("the", "payment")
        assertEquals(expected = "the payment", actual = repairer.repair("the payment"))
    }

    @Test
    fun `repairs a split word inside a sentence`() {
        val repairer = repairerKnowing("the", "apartment", "is", "ready")
        assertEquals(
            expected = "the apartment is ready",
            actual = repairer.repair("the apart ment is ready"),
        )
    }

    @Test
    fun `merges three fragments into one compound`() {
        val repairer = repairerKnowing("understanding")
        assertEquals(expected = "understanding", actual = repairer.repair("under stand ing"))
    }

    @Test
    fun `leaves an unknown standalone token unchanged`() {
        val repairer = repairerKnowing("the", "payment")
        assertEquals(expected = "xyz", actual = repairer.repair("xyz"))
    }

    @Test
    fun `merges fragments that form a typo within edit distance`() {
        val repairer = repairerKnowing("apartment")
        // "apartmnt" is missing one letter (edit distance 1) yet should still be recognized as one word.
        assertEquals(expected = "apartmnt", actual = repairer.repair("apart mnt"))
    }

    @Test
    fun `returns empty string for blank input`() {
        val repairer = repairerKnowing("the")
        assertEquals(expected = "", actual = repairer.repair("   "))
    }
}
