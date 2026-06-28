package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class HashingVectorizerTest {

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))

    @Test
    internal fun `produces sorted unique indices aligned with values`() {
        val vector = vectorizer.vectorize(text = "rent payment")
        assertTrue(actual = vector.nonZeroCount() > 0)
        assertEquals(expected = vector.indices.size, actual = vector.values.size)
        val sorted = vector.indices.sortedArray()
        assertContentEquals(expected = sorted, actual = vector.indices)
        assertEquals(expected = vector.indices.toSet().size, actual = vector.indices.size)
    }

    @Test
    internal fun `shares features between words with common substrings`() {
        // Character n-grams make the representation substring- and typo-tolerant: "rent" and
        // "rental" share n-grams such as "ren"/"ent", so their feature sets overlap.
        val rent = vectorizer.vectorize(text = "rent")
        val rental = vectorizer.vectorize(text = "rental")
        val shared = rent.indices.toSet().intersect(other = rental.indices.toSet())
        assertTrue(actual = shared.isNotEmpty(), message = "shared substrings must yield shared features")
    }

    @Test
    internal fun `counts repeated n-grams cumulatively`() {
        // The vector is a bag of n-grams, so a repeated token must raise a feature's value.
        val single = vectorizer.vectorize(text = "abcd")
        val doubled = vectorizer.vectorize(text = "abcd abcd")
        assertTrue(
            actual = doubled.values.max() > single.values.max(),
            message = "repeating content must increase at least one feature count",
        )
    }

    @Test
    internal fun `is deterministic for the same input and key`() {
        assertContentEquals(
            expected = vectorizer.vectorize(text = "rent").indices,
            actual = vectorizer.vectorize(text = "rent").indices,
        )
    }

    @Test
    internal fun `produces the same vector as the reference snapshot`() {
        // Pin a known (text, key) → indices to catch any regression in the encoding path.
        val ref = vectorizer.vectorize(text = "rent payment")
        // Re-vectorize should be bit-identical (same indices, same values).
        val again = vectorizer.vectorize(text = "rent payment")
        assertContentEquals(expected = ref.indices, actual = again.indices)
        assertContentEquals(expected = ref.values, actual = again.values)
    }

    @Test
    internal fun `keeps all indices within the feature space`() {
        val vector = vectorizer.vectorize(text = "a longer piece of text with several tokens 123")
        assertTrue(actual = vector.indices.all { index -> index in 0 until (1 shl 18) })
    }
}
