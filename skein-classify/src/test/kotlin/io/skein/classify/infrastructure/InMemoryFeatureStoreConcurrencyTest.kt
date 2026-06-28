package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals

internal class InMemoryFeatureStoreConcurrencyTest {

    @Test
    internal fun `concurrent adds from many threads keep every unique observation`() {
        val store = InMemoryFeatureStore()
        val threadCount = 8
        val perThread = 500
        val workers = (0 until threadCount).map { threadIndex ->
            val unique = LabeledFeatures(
                label = Label(value = "label-$threadIndex"),
                features = FeatureVector(indices = intArrayOf(threadIndex), values = floatArrayOf(1.0f)),
            )
            // Each thread adds the same unique observation perThread times — dedup keeps exactly one.
            thread { repeat(times = perThread) { store.add(observation = unique) } }
        }
        workers.forEach { it.join() }
        assertEquals(expected = threadCount, actual = store.size())
    }
}
