package io.skein.classify.infrastructure

/** Fraction of the free heap a training run is allowed to commit to optimiser working memory. */
private const val HEAP_BUDGET_SHARE = 0.5

/** Dense working vectors each minimiser holds beyond its curvature history. */
private const val WORKING_VECTORS_PER_WORKER = 6

/** Vectors per retained curvature pair: the step and the gradient change. */
private const val VECTORS_PER_CURVATURE_PAIR = 2

private const val BYTES_PER_DOUBLE = 8L

/**
 * Chooses how many labels to fit at once.
 *
 * **Why this is not just the core count.** Every concurrent [LbfgsMinimizer] holds its own curvature
 * history of `historySize x 2` vectors of `featureCount` doubles, plus a handful of working
 * vectors. The memory is per worker, so parallelism multiplies it:
 *
 * At the default `historySize = 5`, counting the history and the working vectors together:
 *
 * | features | per worker | x 18 workers |
 * |---|---|---|
 * | 102,205 | 12 MB | 225 MB |
 * | 369,529 | 45 MB | 812 MB |
 * | 4,194,304 | 512 MB | 9.0 GB |
 *
 * Fitting labels in parallel is most of the speed and also exactly how a training run exhausts the
 * heap — and it does so at the widest feature space, which is the configuration a user reaches for
 * when accuracy matters. Defaulting to `parallelStream()`'s common pool makes that failure a
 * function of the machine's core count, which is the one variable that has nothing to do with
 * whether the memory is there.
 */
object TrainingParallelism {

    /**
     * Workers that fit within [heapBudgetBytes], at least one and never more than [maximumWorkers].
     *
     * One worker is always allowed even when the budget says otherwise: refusing to train at all is
     * worse than swapping, and the caller has no other move.
     */
    fun workerCount(
        featureCount: Int,
        historySize: Int,
        heapBudgetBytes: Long = defaultHeapBudget(),
        maximumWorkers: Int = Runtime.getRuntime().availableProcessors(),
    ): Int {
        require(value = featureCount > 0) { "featureCount must be positive, got $featureCount" }
        require(value = historySize > 0) { "historySize must be positive, got $historySize" }
        require(value = maximumWorkers > 0) { "maximumWorkers must be positive, got $maximumWorkers" }
        val perWorker = bytesPerWorker(featureCount = featureCount, historySize = historySize)
        val affordable = (heapBudgetBytes / perWorker).coerceAtMost(maximumValue = maximumWorkers.toLong())
        return affordable.coerceAtLeast(minimumValue = 1L).toInt()
    }

    /** Working memory one minimiser holds while fitting one label over [featureCount] features. */
    fun bytesPerWorker(featureCount: Int, historySize: Int): Long {
        val vectors = historySize.toLong() * VECTORS_PER_CURVATURE_PAIR + WORKING_VECTORS_PER_WORKER
        return vectors * (featureCount.toLong() + 1L) * BYTES_PER_DOUBLE
    }

    /** Half of the heap not currently committed to live objects. */
    private fun defaultHeapBudget(): Long {
        val runtime = Runtime.getRuntime()
        val free = runtime.maxMemory() - runtime.totalMemory() + runtime.freeMemory()
        return (free * HEAP_BUDGET_SHARE).toLong().coerceAtLeast(minimumValue = 1L)
    }
}
