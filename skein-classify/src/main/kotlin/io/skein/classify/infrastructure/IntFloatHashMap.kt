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
        val capacity = tableSizeFor(initialCapacity)
        keys = IntArray(capacity) { EMPTY }
        values = FloatArray(capacity)
        mask = capacity - 1
    }

    fun size(): Int {
        return count
    }

    /** Adds [delta] to the value stored under [key] (inserting `delta` when absent). */
    fun addTo(key: Int, delta: Float) {
        if ((count + 1) * LOAD_FACTOR_NUMERATOR >= keys.size * LOAD_FACTOR_DENOMINATOR) {
            resize(keys.size * 2)
        }
        val slot = slotForInsert(key)
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
        keys.fill(EMPTY)
        count = 0
    }

    /** The entries as parallel arrays, indices sorted ascending (the sparse-vector layout). */
    fun sortedKeysAndValues(): Pair<IntArray, FloatArray> {
        val indices = IntArray(count)
        var position = 0
        for (slot in keys.indices) {
            if (keys[slot] != EMPTY) {
                indices[position] = keys[slot]
                position++
            }
        }
        indices.sort()
        val sortedValues = FloatArray(count)
        for (i in indices.indices) {
            sortedValues[i] = valueOf(indices[i])
        }
        return indices to sortedValues
    }

    private fun slotForInsert(key: Int): Int {
        var slot = spread(key) and mask
        while (keys[slot] != EMPTY && keys[slot] != key) {
            slot = (slot + 1) and mask
        }
        return slot
    }

    private fun valueOf(key: Int): Float {
        var slot = spread(key) and mask
        while (keys[slot] != key) {
            slot = (slot + 1) and mask
        }
        return values[slot]
    }

    private fun resize(requestedCapacity: Int) {
        val oldKeys = keys
        val oldValues = values
        val capacity = tableSizeFor(requestedCapacity)
        keys = IntArray(capacity) { EMPTY }
        values = FloatArray(capacity)
        mask = capacity - 1
        count = 0
        for (slot in oldKeys.indices) {
            if (oldKeys[slot] != EMPTY) {
                addTo(oldKeys[slot], oldValues[slot])
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
    }
}
