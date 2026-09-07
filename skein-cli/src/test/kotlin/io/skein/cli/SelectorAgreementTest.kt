package io.skein.cli

import io.skein.classify.application.ActiveLearningSelector
import io.skein.classify.application.ClassifierKindEnum
import io.skein.classify.domain.Calibration
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.UncertaintyStrategyEnum
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * `PoolSelector` duplicates the uncertainty math of `ActiveLearningSelector` — the CLI needs a
 * variant that scores cached feature vectors in parallel, the library one takes records. The
 * duplication is acknowledged in a `ponytail:` note on `PoolSelector`; this test is what keeps the
 * two honest, by asserting they rank the same records identically rather than by comparing code.
 *
 * It also pins the newer requirement that both consume *calibrated* probabilities: `PoolSelector`
 * bypasses `ClassificationService.classify` for the hot path, so a calibration applied there has to
 * be applied here too or the two would diverge.
 */
internal class SelectorAgreementTest {

    private val schema = Schema.define {
        text(name = "purpose")
        label(name = "category")
    }

    private val candidates = listOf(
        "insurance premium annual policy",
        "rent payment for apartment",
        "insurance rent mixed wording",
        "entirely unrelated wording here",
        "premium apartment insurance rent",
        "monthly policy payment",
    )

    private fun trainedEngine(): CliEngine {
        val engine = CliEngine.fresh(
            schema = schema,
            classifier = ClassifierKindEnum.NAIVE_BAYES,
            hashingConfig = HashingConfig(key0 = 31L, key1 = 17L),
        )
        engine.service.learnAll(
            records = listOf(
                Record(values = mapOf("purpose" to "insurance premium policy", "category" to "INSURANCE")),
                Record(values = mapOf("purpose" to "annual insurance policy", "category" to "INSURANCE")),
                Record(values = mapOf("purpose" to "rent apartment payment", "category" to "RENT")),
                Record(values = mapOf("purpose" to "monthly rent apartment", "category" to "RENT")),
            ),
        )
        return engine
    }

    private fun poolRanking(engine: CliEngine, strategy: UncertaintyStrategyEnum): List<String> {
        val entries = candidates.map { purpose ->
            PoolEntry(row = mutableMapOf<String, Any?>("purpose" to purpose))
        }
        entries.forEach { entry -> entry.features = engine.vectorize(record = Record(values = entry.row)) }
        val selector = PoolSelector(
            engine = engine,
            strategy = strategy,
            scanLimit = 0,
            random = Random(seed = 1L),
        )
        return selector
            .selectMostUncertain(entries = entries, limit = candidates.size)
            .map { entry -> entry.row["purpose"] as String }
    }

    private fun libraryRanking(engine: CliEngine, strategy: UncertaintyStrategyEnum): List<String> {
        return ActiveLearningSelector(service = engine.service)
            .selectForReview(
                candidates = candidates.map { purpose -> Record(values = mapOf("purpose" to purpose)) },
                limit = candidates.size,
                strategy = strategy,
            )
            .map { candidate -> candidate.record["purpose"] as String }
    }

    @Test
    internal fun `both selectors rank identically for every strategy`() {
        val engine = trainedEngine()
        UncertaintyStrategyEnum.entries.forEach { strategy ->
            assertEquals(
                expected = libraryRanking(engine = engine, strategy = strategy),
                actual = poolRanking(engine = engine, strategy = strategy),
                message = "rankings diverged for $strategy",
            )
        }
    }

    @Test
    internal fun `both selectors agree once a calibration is installed`() {
        val engine = trainedEngine()
        engine.service.calibration = Calibration(temperature = 12.0)

        UncertaintyStrategyEnum.entries.forEach { strategy ->
            assertEquals(
                expected = libraryRanking(engine = engine, strategy = strategy),
                actual = poolRanking(engine = engine, strategy = strategy),
                message = "rankings diverged under calibration for $strategy",
            )
        }
    }

    @Test
    internal fun `the engine's cached-vector path returns the calibrated prediction`() {
        val engine = trainedEngine()
        engine.service.calibration = Calibration(temperature = 12.0)
        val record = Record(values = mapOf("purpose" to candidates.first()))

        assertEquals(
            expected = engine.service.classify(record = record).confidence,
            actual = engine.classify(features = engine.vectorize(record = record)).confidence,
            absoluteTolerance = 1e-12,
        )
    }
}
