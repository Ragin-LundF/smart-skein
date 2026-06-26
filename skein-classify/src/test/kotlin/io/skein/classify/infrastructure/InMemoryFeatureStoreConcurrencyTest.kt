package io.skein.classify.infrastructure

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import kotlin.concurrent.thread
import kotlin.test.Test
import kotlin.test.assertEquals

class InMemoryFeatureStoreConcurrencyTest {

    @Test
    fun `concurrent adds from many threads keep every observation`() {
        val store = InMemoryFeatureStore()
        val threadCount = 8
        val perThread = 500
        val observation = LabeledFeatures(
            label = Label("x"),
            features = FeatureVector(indices = intArrayOf(1), values = floatArrayOf(1.0f)),
        )
        val workers = (0 until threadCount).map {
            thread { repeat(perThread) { store.add(observation) } }
        }
        workers.forEach { it.join() }
        assertEquals(expected = threadCount * perThread, actual = store.size())
    }
}
