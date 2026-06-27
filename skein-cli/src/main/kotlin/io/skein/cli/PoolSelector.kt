package io.skein.cli

import io.skein.classify.domain.Prediction
import io.skein.classify.domain.Record
import io.skein.classify.domain.UncertaintyStrategyEnum
import java.util.PriorityQueue
import java.util.stream.IntStream
import kotlin.math.ln
import kotlin.random.Random

/**
 * Picks the most-uncertain entries from a (potentially huge) pool, built for scale:
 *
 * - **Vectorize once.** Each entry's feature vector is cached on first sight ([PoolEntry.features]),
 *   so re-scoring across rounds re-hashes nothing — only a cheap dot product per label.
 * - **Score in parallel.** Vectorizing and scoring fan out across CPU cores; the engine is
 *   thread-safe for concurrent reads, and selection never runs while the model is learning.
 * - **Bounded top-K.** A size-`limit` min-heap replaces a full sort, so memory and time stay
 *   `O(scanned)` and `O(scanned · log limit)` rather than sorting the whole pool.
 * - **Optional sampling.** When [scanLimit] > 0 and the pool is larger, each round scores only a
 *   random window of [scanLimit] entries, making per-round cost independent of pool size. With
 *   [scanLimit] == 0 the whole pool is scanned (exact), which the cache + parallelism keep fast into
 *   the low millions.
 *
 * ponytail: the uncertainty math mirrors `ActiveLearningSelector` (the library's record-in,
 * re-vectorizing variant). Kept in sync by hand — it is a few lines and rarely changes.
 */
class PoolSelector(
    private val engine: CliEngine,
    private val strategy: UncertaintyStrategyEnum,
    private val scanLimit: Int,
    private val random: Random,
) {

    /** Returns up to [limit] entries, most-uncertain first, scoring (a window of) [entries]. */
    fun selectMostUncertain(entries: List<PoolEntry>, limit: Int): List<PoolEntry> {
        if (entries.isEmpty() || limit <= 0) {
            return emptyList()
        }
        val window = scanWindow(poolSize = entries.size, need = limit)
        scoreInParallel(entries = entries, indices = window)
        return topK(entries = entries, indices = window, limit = limit)
    }

    private fun scanWindow(poolSize: Int, need: Int): IntArray {
        val cap = if (scanLimit <= 0) poolSize else maxOf(scanLimit, need)
        if (cap >= poolSize) {
            return IntArray(size = poolSize) { index -> index }
        }
        val chosen = LinkedHashSet<Int>(cap * 2)
        while (chosen.size < cap) {
            chosen.add(random.nextInt(until = poolSize))
        }
        return chosen.toIntArray()
    }

    private fun scoreInParallel(entries: List<PoolEntry>, indices: IntArray) {
        IntStream.range(0, indices.size).parallel().forEach { position ->
            val entry = entries[indices[position]]
            val cached = entry.features
            val vector = if (cached != null) {
                cached
            } else {
                engine.vectorize(record = Record(values = entry.row)).also { computed -> entry.features = computed }
            }
            val prediction = engine.classify(features = vector)
            entry.prediction = prediction
            entry.uncertainty = uncertaintyOf(prediction = prediction)
        }
    }

    private fun topK(entries: List<PoolEntry>, indices: IntArray, limit: Int): List<PoolEntry> {
        val heap = PriorityQueue<PoolEntry>(limit, compareBy { entry -> entry.uncertainty })
        for (index in indices) {
            val entry = entries[index]
            if (heap.size < limit) {
                heap.add(entry)
            } else if (entry.uncertainty > heap.peek().uncertainty) {
                heap.poll()
                heap.add(entry)
            }
        }
        return heap.sortedByDescending { entry -> entry.uncertainty }
    }

    private fun uncertaintyOf(prediction: Prediction): Double {
        return when (strategy) {
            UncertaintyStrategyEnum.MARGIN -> -marginOf(prediction = prediction)
            UncertaintyStrategyEnum.LEAST_CONFIDENCE -> 1.0 - prediction.confidence
            UncertaintyStrategyEnum.ENTROPY -> entropyOf(prediction = prediction)
        }
    }

    private fun marginOf(prediction: Prediction): Double {
        val alternatives = prediction.alternatives
        if (alternatives.size < 2) {
            return 1.0
        }
        return alternatives[0].probability - alternatives[1].probability
    }

    private fun entropyOf(prediction: Prediction): Double {
        var entropy = 0.0
        for (scored in prediction.alternatives) {
            val probability = scored.probability
            if (probability > 0.0) {
                entropy -= probability * ln(x = probability)
            }
        }
        return entropy
    }
}
