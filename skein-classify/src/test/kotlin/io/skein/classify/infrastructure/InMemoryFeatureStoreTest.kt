package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class InMemoryFeatureStoreTest {

    private fun observation(label: String): LabeledFeatures {
        return LabeledFeatures(
            label = Label(label),
            features = FeatureVector(indices = intArrayOf(1, 2), values = floatArrayOf(1.0f, 1.0f)),
        )
    }

    @Test
    fun `stores observations and reports distinct labels`() {
        val store = InMemoryFeatureStore()
        store.addAll(listOf(observation("a"), observation("b"), observation("a")))
        assertEquals(expected = 3, actual = store.size())
        assertEquals(expected = setOf(Label("a"), Label("b")), actual = store.labels())
    }

    @Test
    fun `clear removes all observations`() {
        val store = InMemoryFeatureStore()
        store.add(observation("a"))
        store.clear()
        assertEquals(expected = 0, actual = store.size())
        assertTrue(store.all().isEmpty())
    }
}
