package io.skein.classify.embedding.onnx.application

import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

internal class EmbeddingCacheTest {

    @Test
    internal fun `returns what it was given`() {
        val cache = EmbeddingCache()

        cache.put(key = "a", embedding = floatArrayOf(1.0f, 2.0f))

        assertEquals(expected = listOf(1.0f, 2.0f), actual = cache.get(key = "a")!!.toList())
        assertNull(actual = cache.get(key = "missing"))
    }

    /** Bounded on purpose: a cache that grows with the corpus defeats the point at a million rows. */
    @Test
    internal fun `never holds more than its bound`() {
        val cache = EmbeddingCache(maxEntries = 3)

        repeat(times = 10) { index -> cache.put(key = "k$index", embedding = floatArrayOf(index.toFloat())) }

        assertEquals(expected = 3, actual = cache.size())
    }

    @Test
    internal fun `evicts the least recently used entry`() {
        val cache = EmbeddingCache(maxEntries = 2)
        cache.put(key = "a", embedding = floatArrayOf(1.0f))
        cache.put(key = "b", embedding = floatArrayOf(2.0f))

        // Touching "a" makes "b" the least recently used, so the next insert evicts "b", not "a".
        cache.get(key = "a")
        cache.put(key = "c", embedding = floatArrayOf(3.0f))

        assertNotNull(actual = cache.get(key = "a"))
        assertNull(actual = cache.get(key = "b"))
        assertNotNull(actual = cache.get(key = "c"))
    }

    @Test
    internal fun `overwrites an existing key without growing`() {
        val cache = EmbeddingCache(maxEntries = 2)

        cache.put(key = "a", embedding = floatArrayOf(1.0f))
        cache.put(key = "a", embedding = floatArrayOf(9.0f))

        assertEquals(expected = 1, actual = cache.size())
        assertEquals(expected = 9.0f, actual = cache.get(key = "a")!![0])
    }

    @Test
    internal fun `clears`() {
        val cache = EmbeddingCache()
        cache.put(key = "a", embedding = floatArrayOf(1.0f))

        cache.clear()

        assertEquals(expected = 0, actual = cache.size())
    }

    /**
     * A training run embeds batches from several threads, and `LinkedHashMap` in access order
     * mutates its links on a plain `get` — an unsynchronized read is a writer.
     */
    @Test
    internal fun `survives concurrent readers and writers`() {
        val cache = EmbeddingCache(maxEntries = 64)
        val pool = Executors.newFixedThreadPool(8)
        val failures = java.util.concurrent.atomic.AtomicInteger()

        repeat(times = 8) { worker ->
            pool.execute {
                runCatching {
                    repeat(times = 2_000) { index ->
                        cache.put(key = "w$worker-$index", embedding = floatArrayOf(index.toFloat()))
                        cache.get(key = "w$worker-${index / 2}")
                    }
                }.onFailure { failures.incrementAndGet() }
            }
        }
        pool.shutdown()

        assertTrue(actual = pool.awaitTermination(30, TimeUnit.SECONDS))
        assertEquals(expected = 0, actual = failures.get())
        assertTrue(actual = cache.size() <= 64, message = "cache grew past its bound: ${cache.size()}")
    }

    @Test
    internal fun `rejects a non-positive bound`() {
        assertFailsWith<IllegalArgumentException> { EmbeddingCache(maxEntries = 0) }
    }
}
