package io.skein.store.postgres.infrastructure

import io.skein.classify.domain.FeatureVector
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

class FeatureVectorCodecTest {

    private val codec = FeatureVectorCodec()

    @Test
    fun `round-trips a sparse vector exactly`() {
        val original = FeatureVector(indices = intArrayOf(3, 17, 42), values = floatArrayOf(1.0f, 2.5f, 0.25f))
        val decoded = codec.decode(codec.encode(original))
        assertContentEquals(expected = original.indices, actual = decoded.indices)
        assertContentEquals(expected = original.values, actual = decoded.values)
    }

    @Test
    fun `round-trips an empty vector`() {
        val decoded = codec.decode(codec.encode(FeatureVector(indices = intArrayOf(), values = floatArrayOf())))
        assertEquals(expected = 0, actual = decoded.nonZeroCount())
    }
}
