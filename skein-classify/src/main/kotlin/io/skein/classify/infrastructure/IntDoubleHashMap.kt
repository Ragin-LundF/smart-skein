package io.skein.classify.infrastructure

/**
 * Primitive open-addressing (linear-probe) `int → double` map for O(1) feature score lookups
 * without boxing. Keys must be non-negative (feature indices); `-1` is the empty sentinel.
 *
 * Designed as a read-heavy lookup structure: [put] all entries at construction time, then
 * call [get] on the hot path. No resize — callers must supply adequate initial capacity.
 */
class IntDoubleHashMap(initialCapacity: Int) {

    private val keys: IntArray
    private val values: DoubleArray
    private val mask: Int

    init {
        val capacity = tableSizeFor(requested = initialCapacity)
        keys = IntArray(size = capacity) { EMPTY }
        values = DoubleArray(size = capacity)
        mask = capacity - 1
    }

    fun put(key: Int, value: Double) {
        var slot = spread(key = key) and mask
        while (keys[slot] != EMPTY && keys[slot] != key) {
            slot = (slot + 1) and mask
        }
        keys[slot] = key
        values[slot] = value
    }

    fun get(key: Int): Double {
        var slot = spread(key = key) and mask
        while (true) {
            val k = keys[slot]
            if (k == EMPTY) return 0.0
            if (k == key) return values[slot]
            slot = (slot + 1) and mask
        }
    }

    private fun spread(key: Int): Int {
        return key xor (key ushr SPREAD_SHIFT)
    }

    private fun tableSizeFor(requested: Int): Int {
        var size = MIN_CAPACITY
        while (size < requested) {
            size = size shl 1
        }
        return size
    }

    private companion object {
        const val MIN_CAPACITY = 16
        const val EMPTY = -1
        const val SPREAD_SHIFT = 16
    }
}
