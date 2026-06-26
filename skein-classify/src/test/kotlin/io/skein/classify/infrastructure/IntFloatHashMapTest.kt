package io.skein.classify.infrastructure

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals

internal class IntFloatHashMapTest {

    @Test
    internal fun `accumulates repeated keys and emits sorted entries`() {
        val map = IntFloatHashMap()
        map.addTo(key = 5, delta = 1.0f)
        map.addTo(key = 1, delta = 1.0f)
        map.addTo(key = 5, delta = 2.0f)
        val (indices, values) = map.sortedKeysAndValues()
        assertContentEquals(expected = intArrayOf(1, 5), actual = indices)
        assertContentEquals(expected = floatArrayOf(1.0f, 3.0f), actual = values)
        assertEquals(expected = 2, actual = map.size())
    }

    @Test
    internal fun `grows past its initial capacity without losing entries`() {
        val map = IntFloatHashMap(initialCapacity = 4)
        for (key in 0 until 100) {
            map.addTo(key = key, delta = 1.0f)
        }
        val (indices, values) = map.sortedKeysAndValues()
        assertEquals(expected = 100, actual = map.size())
        assertContentEquals(expected = IntArray(size = 100) { it }, actual = indices)
        assertContentEquals(expected = FloatArray(size = 100) { 1.0f }, actual = values)
    }

    @Test
    internal fun `clear resets the map for reuse`() {
        val map = IntFloatHashMap()
        map.addTo(key = 7, delta = 4.0f)
        map.clear()
        assertEquals(expected = 0, actual = map.size())
        map.addTo(key = 9, delta = 1.0f)
        val (indices, _) = map.sortedKeysAndValues()
        assertContentEquals(expected = intArrayOf(9), actual = indices)
    }
}
