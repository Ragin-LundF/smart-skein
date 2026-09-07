package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

/**
 * Pins the exact hashed output of [HashingVectorizer] for a fixed key and input.
 *
 * These literals were generated from the implementation and must not be "fixed" if they start
 * failing: the bucket for an n-gram is part of the on-disk model format, so a change here silently
 * invalidates every previously saved `.skein` file. A failure means the feature extraction changed,
 * which is a breaking change to make deliberately, not incidentally.
 */
internal class HashingVectorizerGoldenTest {

    private val goldenIndices = intArrayOf(
        5865, 10085, 16869, 17520, 18445, 20211, 20271, 20806, 23145, 26014,
        26377, 35286, 44376, 44893, 50461, 54940, 62410, 65644, 66197, 66461,
        67314, 73787, 81214, 81407, 82334, 85265, 88689, 90009, 96406, 97403,
        103754, 106010, 106905, 112312, 126575, 127974, 129678, 133353, 133365, 133521,
        135171, 135254, 135739, 137113, 137164, 141531, 148722, 151108, 151250, 155996,
        159455, 161481, 162524, 164579, 170637, 170908, 172149, 172834, 172902, 175182,
        176383, 179554, 182959, 183006, 184674, 184772, 186538, 190833, 191765, 192749,
        195830, 197765, 198026, 202689, 208755, 216559, 220014, 223304, 225491, 227653,
        240008, 240049, 244974, 256148, 257482, 259498, 260970,
    )

    private val vectorizer = HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L))

    @Test
    internal fun `produces the pinned buckets for a fixed key and text`() {
        val vector = vectorizer.vectorize(text = "rent transfer landlord monthly")

        assertContentEquals(expected = goldenIndices, actual = vector.indices)
        assertEquals(expected = goldenIndices.size, actual = vector.values.size)
    }

    @Test
    internal fun `produces the pinned counts, including the one repeated n-gram`() {
        val vector = vectorizer.vectorize(text = "rent transfer landlord monthly")

        // Exactly one n-gram occurs twice in this text; every other bucket is hit once.
        assertEquals(expected = 2.0f, actual = vector.values[85])
        vector.values.forEachIndexed { position, value ->
            val expected = if (position == 85) 2.0f else 1.0f
            assertEquals(expected = expected, actual = value, message = "bucket at position $position")
        }
    }

    @Test
    internal fun `indices are sorted ascending and distinct`() {
        val vector = vectorizer.vectorize(text = "rent transfer landlord monthly")

        assertContentEquals(expected = vector.indices.sortedArray(), actual = vector.indices)
        assertEquals(expected = vector.indices.size, actual = vector.indices.toSet().size)
    }
}
