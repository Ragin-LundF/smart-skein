package io.skein.classify.application

import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.PrivacyModeEnum
import io.skein.classify.domain.Record
import io.skein.classify.domain.Schema
import io.skein.classify.domain.UncertaintyStrategyEnum
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ActiveLearningSelectorTest {

    private val schema = Schema.define {
        text("purpose")
        label("category")
    }

    private fun record(purpose: String): Record {
        return Record(mapOf("purpose" to purpose))
    }

    private fun trainedService(): ClassificationService {
        val service = ClassificationService(
            schema = schema,
            privacyMode = PrivacyModeEnum.FEATURES_ONLY,
            hashingConfig = HashingConfig(key0 = 1L, key1 = 2L),
        )
        service.learnAll(
            listOf(
                Record(mapOf("purpose" to "rent transfer landlord", "category" to "housing")),
                Record(mapOf("purpose" to "monthly apartment rent", "category" to "housing")),
                Record(mapOf("purpose" to "salary october payout", "category" to "income")),
                Record(mapOf("purpose" to "monthly salary payment", "category" to "income")),
            ),
        )
        return service
    }

    @Test
    fun `ranks the most uncertain record first`() {
        val selector = ActiveLearningSelector(trainedService())
        val confident = record("monthly apartment rent")
        val uncertain = record("unrelated neutral wording xyz")

        val selected = selector.selectForReview(listOf(confident, uncertain), limit = 2)

        assertEquals(expected = 2, actual = selected.size)
        assertEquals(
            expected = uncertain,
            actual = selected.first().record,
            message = "smallest top-two margin must come first",
        )
        assertTrue(selected[0].margin <= selected[1].margin)
    }

    @Test
    fun `respects the limit`() {
        val selector = ActiveLearningSelector(trainedService())
        val selected = selector.selectForReview(
            listOf(record("a"), record("b"), record("c")),
            limit = 1,
        )
        assertEquals(expected = 1, actual = selected.size)
    }

    @Test
    fun `least-confidence and entropy strategies also rank the uncertain record first`() {
        val selector = ActiveLearningSelector(trainedService())
        val confident = record("monthly apartment rent")
        val uncertain = record("unrelated neutral wording xyz")
        val pool = listOf(confident, uncertain)

        val byLeastConfidence = selector.selectForReview(
            pool,
            limit = 2,
            strategy = UncertaintyStrategyEnum.LEAST_CONFIDENCE,
        )
        val byEntropy = selector.selectForReview(pool, limit = 2, strategy = UncertaintyStrategyEnum.ENTROPY)

        assertEquals(expected = uncertain, actual = byLeastConfidence.first().record)
        assertEquals(expected = uncertain, actual = byEntropy.first().record)
    }
}
