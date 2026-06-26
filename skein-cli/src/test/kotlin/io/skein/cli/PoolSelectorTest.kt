package io.skein.cli

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.UncertaintyStrategyEnum
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

internal class PoolSelectorTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }

    private fun trainedEngine(): CliEngine {
        val engine = CliEngine.fresh(
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = HashingConfig(key0 = 3L, key1 = 9L),
        )
        engine.service.learnAll(
            records = listOf(
                Record(values = mapOf("purpose" to "insurance premium policy", "category" to "insurance")),
                Record(values = mapOf("purpose" to "rent apartment payment", "category" to "rent")),
            ),
        )
        return engine
    }

    private fun pool(vararg purposes: String): List<PoolEntry> {
        return purposes.map { purpose -> PoolEntry(row = linkedMapOf("purpose" to purpose, "category" to "")) }
    }

    @Test
    internal fun `scores caches the feature vector and returns up to limit entries, most uncertain first`() {
        val selector = PoolSelector(
            engine = trainedEngine(),
            strategy = UncertaintyStrategyEnum.MARGIN,
            scanLimit = 0,
            random = Random(seed = 1),
        )
        val entries = pool("insurance premium policy", "totally unrelated mystery text", "rent apartment payment")

        val picks = selector.selectMostUncertain(entries = entries, limit = 2)

        assertEquals(expected = 2, actual = picks.size)
        // Most-uncertain first: each pick is at least as uncertain as the next.
        assertTrue(actual = picks[0].uncertainty >= picks[1].uncertainty)
        // Every scanned entry vectorized exactly once and cached.
        assertTrue(actual = entries.all { entry -> entry.features != null }, message = "all vectors cached")
        assertNotNull(actual = picks[0].prediction)
    }

    @Test
    internal fun `scan-limit bounds the number of entries scored per round`() {
        val selector = PoolSelector(
            engine = trainedEngine(),
            strategy = UncertaintyStrategyEnum.MARGIN,
            scanLimit = 2,
            random = Random(seed = 42),
        )
        val entries = pool("a text", "b text", "c text", "d text", "e text")

        selector.selectMostUncertain(entries = entries, limit = 1)

        // With scanLimit 2 only a sampled window is vectorized, not the whole pool.
        assertEquals(expected = 2, actual = entries.count { entry -> entry.features != null })
    }
}
