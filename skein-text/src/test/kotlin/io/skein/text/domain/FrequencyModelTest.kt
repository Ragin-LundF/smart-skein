package io.skein.text.domain

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class FrequencyModelTest {

    @Test
    fun `counts occurrences case-insensitively`() {
        val model = FrequencyModel()
        model.learnAll(listOf("Apartment", "apartment", "APARTMENT"))
        assertEquals(expected = 3, actual = model.frequency("apartment"))
    }

    @Test
    fun `ignores blank input`() {
        val model = FrequencyModel()
        model.learn("   ")
        assertTrue(model.knownWords().isEmpty())
    }

    @Test
    fun `hides words below the privacy threshold`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learn("john")
        assertFalse(model.isKnown("john"), message = "seen once must stay below threshold 2")
        assertEquals(expected = 0, actual = model.frequency("john"))
        model.learn("john")
        assertTrue(model.isKnown("john"), message = "seen twice must reach threshold 2")
        assertEquals(expected = 2, actual = model.frequency("john"))
    }

    @Test
    fun `knownWords excludes rare words`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learnAll(listOf("the", "the", "smith"))
        assertEquals(expected = setOf("the"), actual = model.knownWords())
    }

    @Test
    fun `serialization round-trips known words and drops rare ones`() {
        val model = FrequencyModel(minKeepFrequency = 2)
        model.learnAll(listOf("the", "the", "the", "apartment", "apartment", "smith"))

        val restored = FrequencyModel.deserialize(model.serialize())

        assertEquals(expected = 3, actual = restored.frequency("the"))
        assertEquals(expected = 2, actual = restored.frequency("apartment"))
        assertEquals(expected = setOf("the", "apartment"), actual = restored.knownWords())
        // "smith" was sub-threshold, so it was never serialized.
        assertFalse(restored.isKnown("smith"))
    }
}
