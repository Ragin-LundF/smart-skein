package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class HashingVectorizerTest {

    private val vectorizer = HashingVectorizer(HashingConfig(key0 = 1L, key1 = 2L))

    @Test
    fun `produces sorted unique indices aligned with values`() {
        val vector = vectorizer.vectorize("rent payment")
        assertTrue(vector.nonZeroCount() > 0)
        assertEquals(expected = vector.indices.size, actual = vector.values.size)
        val sorted = vector.indices.sortedArray()
        assertContentEquals(expected = sorted, actual = vector.indices)
        assertEquals(expected = vector.indices.toSet().size, actual = vector.indices.size)
    }

    @Test
    fun `shares features between words with common substrings`() {
        // Character n-grams make the representation substring- and typo-tolerant: "rent" and
        // "rental" share n-grams such as "ren"/"ent", so their feature sets overlap.
        val rent = vectorizer.vectorize("rent")
        val rental = vectorizer.vectorize("rental")
        val shared = rent.indices.toSet().intersect(rental.indices.toSet())
        assertTrue(shared.isNotEmpty(), message = "shared substrings must yield shared features")
    }

    @Test
    fun `counts repeated n-grams cumulatively`() {
        // The vector is a bag of n-grams, so a repeated token must raise a feature's value.
        val single = vectorizer.vectorize("abcd")
        val doubled = vectorizer.vectorize("abcd abcd")
        assertTrue(
            doubled.values.max() > single.values.max(),
            message = "repeating content must increase at least one feature count",
        )
    }

    @Test
    fun `is deterministic for the same input and key`() {
        assertContentEquals(
            expected = vectorizer.vectorize("rent").indices,
            actual = vectorizer.vectorize("rent").indices,
        )
    }

    @Test
    fun `keeps all indices within the feature space`() {
        val vector = vectorizer.vectorize("a longer piece of text with several tokens 123")
        assertTrue(vector.indices.all { index -> index in 0 until (1 shl 18) })
    }
}
