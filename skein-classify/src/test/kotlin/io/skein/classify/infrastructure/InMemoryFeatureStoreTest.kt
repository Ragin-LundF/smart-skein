package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

internal class InMemoryFeatureStoreTest {

    private fun observation(label: String): LabeledFeatures {
        return LabeledFeatures(
            label = Label(value = label),
            features = FeatureVector(indices = intArrayOf(1, 2), values = floatArrayOf(1.0f, 1.0f)),
        )
    }

    @Test
    internal fun `stores observations and reports distinct labels`() {
        val store = InMemoryFeatureStore()
        store.addAll(observations = listOf(observation(label = "a"), observation(label = "b")))
        assertEquals(expected = 2, actual = store.size())
        assertEquals(expected = setOf(Label(value = "a"), Label(value = "b")), actual = store.labels())
    }

    @Test
    internal fun `duplicate observations are silently discarded`() {
        val store = InMemoryFeatureStore()
        store.addAll(
            observations = listOf(observation(label = "a"), observation(label = "b"), observation(label = "a")),
        )
        assertEquals(expected = 2, actual = store.size())
    }

    @Test
    internal fun `same vector under different labels counts as two distinct observations`() {
        val store = InMemoryFeatureStore()
        store.add(observation(label = "x"))
        store.add(observation(label = "y"))
        assertEquals(expected = 2, actual = store.size())
    }

    @Test
    internal fun `clear resets deduplication so the same observation can be re-added`() {
        val store = InMemoryFeatureStore()
        store.add(observation(label = "a"))
        store.clear()
        store.add(observation(label = "a"))
        assertEquals(expected = 1, actual = store.size())
    }

    @Test
    internal fun `clear removes all observations`() {
        val store = InMemoryFeatureStore()
        store.add(observation = observation(label = "a"))
        store.clear()
        assertEquals(expected = 0, actual = store.size())
        assertTrue(actual = store.all().isEmpty())
    }
}
