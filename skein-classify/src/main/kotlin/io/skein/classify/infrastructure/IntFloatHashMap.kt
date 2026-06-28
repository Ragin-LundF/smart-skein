package io.skein.classify.infrastructure

/**
 * Primitive open-addressing (linear-probe) `int → float` map for accumulating sparse feature counts
 * without boxing. Keys must be **non-negative** (feature indices); `-1` is the empty sentinel.
 *
 * Designed to be cleared and reused across calls (e.g. a per-thread scratch buffer), avoiding the
 * per-call allocation a boxed `HashMap<Int, Float>` would incur on the vectorization hot path.
 */
class IntFloatHashMap(initialCapacity: Int = DEFAULT_CAPACITY) {

    private var keys: IntArray
    private var values: FloatArray
    private var mask: Int
    private var count: Int = 0

    init {
        val capacity = tableSizeFor(requested = initialCapacity)
        keys = IntArray(size = capacity) { EMPTY }
        values = FloatArray(size = capacity)
        mask = capacity - 1
    }

    fun size(): Int {
        return count
    }

    /** Adds [delta] to the value stored under [key] (inserting `delta` when absent). */
    fun addTo(key: Int, delta: Float) {
        if ((count + 1) * LOAD_FACTOR_NUMERATOR >= keys.size * LOAD_FACTOR_DENOMINATOR) {
            resize(requestedCapacity = keys.size * 2)
        }
        val slot = slotForInsert(key = key)
        if (keys[slot] == EMPTY) {
            keys[slot] = key
            values[slot] = delta
            count++
        } else {
            values[slot] += delta
        }
    }

    /** Resets the map to empty for reuse, keeping the allocated backing arrays. */
    fun clear() {
        keys.fill(element = EMPTY)
        count = 0
    }

    /** The entries as parallel arrays, indices sorted ascending (the sparse-vector layout). */
    fun sortedKeysAndValues(): Pair<IntArray, FloatArray> {
        // Pack each (key, value) into one Long so a single sort handles both arrays together,
        // avoiding a second linear-probe pass to look up values by sorted key.
        val packed = LongArray(size = count)
        var position = 0
        for (slot in keys.indices) {
            if (keys[slot] != EMPTY) {
                packed[position++] = (keys[slot].toLong() shl INT_BITS) or
                    (java.lang.Float.floatToRawIntBits(values[slot]).toLong() and LOW_INT_MASK)
            }
        }
        packed.sort()
        val outIndices = IntArray(size = count)
        val outValues = FloatArray(size = count)
        for (i in packed.indices) {
            outIndices[i] = (packed[i] ushr INT_BITS).toInt()
            outValues[i] = java.lang.Float.intBitsToFloat((packed[i] and LOW_INT_MASK).toInt())
        }
        return outIndices to outValues
    }

    private fun slotForInsert(key: Int): Int {
        var slot = spread(key = key) and mask
        while (keys[slot] != EMPTY && keys[slot] != key) {
            slot = (slot + 1) and mask
        }
        return slot
    }

    private fun resize(requestedCapacity: Int) {
        val oldKeys = keys
        val oldValues = values
        val capacity = tableSizeFor(requested = requestedCapacity)
        keys = IntArray(size = capacity) { EMPTY }
        values = FloatArray(size = capacity)
        mask = capacity - 1
        count = 0
        for (slot in oldKeys.indices) {
            if (oldKeys[slot] != EMPTY) {
                addTo(key = oldKeys[slot], delta = oldValues[slot])
            }
        }
    }

    private fun spread(key: Int): Int {
        return key xor (key ushr SPREAD_SHIFT)
    }

    private fun tableSizeFor(requested: Int): Int {
        var size = DEFAULT_CAPACITY
        while (size < requested) {
            size = size shl 1
        }
        return size
    }

    private companion object {
        const val DEFAULT_CAPACITY = 16
        const val EMPTY = -1
        const val SPREAD_SHIFT = 16
        const val LOAD_FACTOR_NUMERATOR = 4
        const val LOAD_FACTOR_DENOMINATOR = 3
        const val INT_BITS = Int.SIZE_BITS          // bits to shift an Int into the high half of a Long
        const val LOW_INT_MASK = 0xFFFFFFFFL        // mask for the low Int-width bits of a Long
    }
}
