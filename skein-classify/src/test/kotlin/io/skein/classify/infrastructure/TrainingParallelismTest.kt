package io.skein.classify.infrastructure

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class TrainingParallelismTest {

    @Test
    internal fun `bounds workers by the heap budget rather than the core count`() {
        val perWorker = TrainingParallelism.bytesPerWorker(featureCount = 100_000, historySize = 5)

        val workers = TrainingParallelism.workerCount(
            featureCount = 100_000,
            historySize = 5,
            heapBudgetBytes = perWorker * 3,
            maximumWorkers = 64,
        )

        assertEquals(expected = 3, actual = workers)
    }

    @Test
    internal fun `never exceeds the caller's worker ceiling even with a huge budget`() {
        val workers = TrainingParallelism.workerCount(
            featureCount = 1_000,
            historySize = 5,
            heapBudgetBytes = Long.MAX_VALUE / 2,
            maximumWorkers = 4,
        )

        assertEquals(expected = 4, actual = workers)
    }

    /** Refusing to train is worse than training slowly, so one worker is always permitted. */
    @Test
    internal fun `always allows a single worker even when the budget cannot afford one`() {
        val workers = TrainingParallelism.workerCount(
            featureCount = 4_194_304,
            historySize = 20,
            heapBudgetBytes = 1L,
            maximumWorkers = 32,
        )

        assertEquals(expected = 1, actual = workers)
    }

    @Test
    internal fun `working memory grows with both the feature count and the history size`() {
        val narrow = TrainingParallelism.bytesPerWorker(featureCount = 1_000, historySize = 5)
        val wide = TrainingParallelism.bytesPerWorker(featureCount = 100_000, historySize = 5)
        val deep = TrainingParallelism.bytesPerWorker(featureCount = 1_000, historySize = 20)

        assertTrue(actual = wide > narrow * 50)
        assertTrue(actual = deep > narrow)
    }

    /** The table in the KDoc, made checkable, so the documented budget cannot drift from the code. */
    @Test
    internal fun `matches the documented per-worker figures`() {
        val megabyte = 1024L * 1024L

        assertEquals(
            expected = 12L,
            actual = TrainingParallelism.bytesPerWorker(featureCount = 102_205, historySize = 5) / megabyte,
        )
        assertEquals(
            expected = 45L,
            actual = TrainingParallelism.bytesPerWorker(featureCount = 369_529, historySize = 5) / megabyte,
        )
        assertEquals(
            expected = 512L,
            actual = TrainingParallelism.bytesPerWorker(featureCount = 4_194_304, historySize = 5) / megabyte,
        )
    }

    @Test
    internal fun `rejects a non-positive configuration`() {
        assertFailsWith<IllegalArgumentException> {
            TrainingParallelism.workerCount(featureCount = 0, historySize = 5)
        }
        assertFailsWith<IllegalArgumentException> {
            TrainingParallelism.workerCount(featureCount = 10, historySize = 0)
        }
        assertFailsWith<IllegalArgumentException> {
            TrainingParallelism.workerCount(featureCount = 10, historySize = 5, maximumWorkers = 0)
        }
    }
}
