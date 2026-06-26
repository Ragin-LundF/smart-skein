package io.skein.text.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

internal class FrequencyModelTest {

    @Test
    internal fun `counts occurrences case-insensitively`() {
        val model = FrequencyModel()
        model.learnAll(words = listOf("Apartment", "apartment", "APARTMENT"))
        assertEquals(expected = 3, actual = model.frequency(word = "apartment"))
    }

    @Test
    internal fun `ignores blank input`() {
        val model = FrequencyModel()
        model.learn(word = "   ")
        assertTrue(actual = model.knownWords().isEmpty())
    }

    @Test
    internal fun `hides words below the privacy threshold`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learn(word = "john")
        assertFalse(actual = model.isKnown(word = "john"), message = "seen once must stay below threshold 2")
        assertEquals(expected = 0, actual = model.frequency(word = "john"))
        model.learn(word = "john")
        assertTrue(actual = model.isKnown(word = "john"), message = "seen twice must reach threshold 2")
        assertEquals(expected = 2, actual = model.frequency(word = "john"))
    }

    @Test
    internal fun `knownWords excludes rare words`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learnAll(words = listOf("the", "the", "smith"))
        assertEquals(expected = setOf("the"), actual = model.knownWords())
    }

    @Test
    internal fun `serialization round-trips known words and drops rare ones`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learnAll(words = listOf("the", "the", "the", "apartment", "apartment", "smith"))

        val restored = FrequencyModel.deserialize(serialized = model.serialize())

        assertEquals(expected = 3, actual = restored.frequency(word = "the"))
        assertEquals(expected = 2, actual = restored.frequency(word = "apartment"))
        assertEquals(expected = setOf("the", "apartment"), actual = restored.knownWords())
        // "smith" was sub-threshold, so it was never serialized.
        assertFalse(actual = restored.isKnown(word = "smith"))
    }
}
