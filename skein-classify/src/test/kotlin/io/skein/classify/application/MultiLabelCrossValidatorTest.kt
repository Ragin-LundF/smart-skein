package io.skein.classify.application

import io.skein.classify.domain.FeatureVector
import io.skein.classify.domain.HashingConfig
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledText
import io.skein.classify.domain.VectorizerFingerprint
import io.skein.classify.infrastructure.LbfgsMultiLabelLearner
import io.skein.classify.spi.BatchLearner
import io.skein.classify.spi.BatchVectorizer
import io.skein.classify.spi.Vectorizer
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

internal class MultiLabelCrossValidatorTest {

    private val featureCount = 1 shl 14
    private val validator = MultiLabelCrossValidator()

    private fun vectorizer(): Vectorizer {
        return HashingVectorizer(config = HashingConfig(key0 = 1L, key1 = 2L, numFeatures = featureCount))
    }

    private fun learner(): BatchLearner {
        return LbfgsMultiLabelLearner(
            featureCount = featureCount,
            inverseRegularization = 10.0,
            gradientTolerance = 1e-5,
            keepFraction = 1.0,
        )
    }

    /**
     * Twenty originating rules, five examples each, and the label depends on nothing but the rule's
     * own keyword.
     *
     * The keywords are unrelated invented words rather than `keyword0`, `keyword1`, ... on purpose.
     * Systematically named keywords share character n-grams that *genuinely* predict the label, so
     * a model generalises across rules honestly and a grouped split correctly scores well -- which
     * measures something real, but not the leak this fixture is for. With unrelated tokens there is
     * nothing to carry from one rule to another, so any score above chance on a grouped split could
     * only have come from a sibling on the wrong side.
     */
    private fun ruleGeneratedCorpus(grouped: Boolean): List<MultiLabeledText> {
        return KEYWORDS.entries.toList().flatMapIndexed { rule, (keyword, label) ->
            (0 until 5).map { example ->
                MultiLabeledText(
                    featureText = "$keyword example$example filler text",
                    labels = setOf(Label(value = label)),
                    group = if (grouped) "rule-$rule" else null,
                )
            }
        }
    }

    private companion object {
        /** Twenty unrelated tokens, split ten/ten between two labels in no discernible pattern. */
        val KEYWORDS = mapOf(
            "qufrel" to "LEFT", "paznit" to "RIGHT", "vobrem" to "RIGHT", "luntak" to "LEFT",
            "zephyd" to "RIGHT", "mirlow" to "LEFT", "gadrun" to "RIGHT", "teskov" to "LEFT",
            "broqim" to "RIGHT", "yanvel" to "LEFT", "chodra" to "RIGHT", "pimsuk" to "LEFT",
            "reflan" to "RIGHT", "xotibe" to "LEFT", "junwar" to "RIGHT", "delpho" to "LEFT",
            "kravis" to "RIGHT", "nurbet" to "LEFT", "oswiln" to "RIGHT", "fatchy" to "LEFT",
        )
    }

    /**
     * The measurement the plan is built on, reproduced in miniature.
     *
     * With the rows grouped by originating rule, every held-out example comes from a rule the model
     * never saw, and it scores badly because there genuinely is nothing to generalise from. Ungroup
     * the same rows and siblings land on both sides of the split, the model reads the keyword off
     * the training set, and the score jumps. The grouped number is the honest one; the difference
     * is exactly the inflation a random split buys.
     */
    @Test
    internal fun `grouping by origin scores lower than splitting siblings apart`() {
        val grouped = validator.crossValidate(
            corpus = ruleGeneratedCorpus(grouped = true),
            vectorizerFactory = { vectorizer() },
            learnerFactory = { learner() },
            folds = 5,
        )
        val ungrouped = validator.crossValidate(
            corpus = ruleGeneratedCorpus(grouped = false),
            vectorizerFactory = { vectorizer() },
            learnerFactory = { learner() },
            folds = 5,
        )

        assertTrue(
            actual = ungrouped.pooled.micro.f1 > grouped.pooled.micro.f1,
            message = "ungrouped ${ungrouped.pooled.micro.f1} did not beat grouped ${grouped.pooled.micro.f1}",
        )
        assertTrue(
            actual = ungrouped.pooled.micro.recall > 0.9,
            message = "leaked split should recall almost everything, got ${ungrouped.pooled.micro.recall}",
        )
    }

    /**
     * With feature hashing there is no fitted state, so this cannot fail today — which is precisely
     * why it is pinned now. The moment an IDF table or a vocabulary is introduced, a factory called
     * once over the whole corpus leaks held-out statistics into every fold's features and every
     * score afterwards is wrong in the flattering direction.
     */
    @Test
    internal fun `builds a vectorizer per fold from that fold's training rows only`() {
        val corpus = ruleGeneratedCorpus(grouped = true)
        val fittedOn = ArrayList<Int>()

        validator.crossValidate(
            corpus = corpus,
            vectorizerFactory = { training ->
                fittedOn.add(element = training.size)
                vectorizer()
            },
            learnerFactory = { learner() },
            folds = 5,
        )

        assertEquals(expected = 5, actual = fittedOn.size)
        assertTrue(
            actual = fittedOn.all { size -> size < corpus.size },
            message = "a vectorizer saw the whole corpus: $fittedOn",
        )
        assertEquals(expected = corpus.size * 4, actual = fittedOn.sum())
    }

    @Test
    internal fun `holds out every row exactly once across the run`() {
        val corpus = ruleGeneratedCorpus(grouped = true)

        val report = validator.crossValidate(
            corpus = corpus,
            vectorizerFactory = { vectorizer() },
            learnerFactory = { learner() },
            folds = 5,
        )

        assertEquals(expected = corpus.size, actual = report.pooledOutcomes.size)
        assertEquals(expected = corpus.size, actual = report.pooled.sampleCount)
        assertEquals(expected = 5, actual = report.folds.size)
        assertEquals(expected = corpus.size, actual = report.folds.sumOf { fold -> fold.sampleCount })
    }

    @Test
    internal fun `keeps the pooled scores so a threshold sweep needs no refitting`() {
        val report = validator.crossValidate(
            corpus = ruleGeneratedCorpus(grouped = false),
            vectorizerFactory = { vectorizer() },
            learnerFactory = { learner() },
            folds = 5,
        )

        val sweep = MultiLabelEvaluator().sweep(outcomes = report.pooledOutcomes)

        assertEquals(expected = 7, actual = sweep.size)
        assertTrue(actual = sweep.zipWithNext().all { (low, high) -> low.recall >= high.recall })
    }

    @Test
    internal fun `reports per-fold variance alongside the mean`() {
        val report = validator.crossValidate(
            corpus = ruleGeneratedCorpus(grouped = false),
            vectorizerFactory = { vectorizer() },
            learnerFactory = { learner() },
            folds = 5,
        )

        assertTrue(actual = report.meanMicroF1() > 0.0)
        assertTrue(actual = report.meanMacroF1() > 0.0)
        assertTrue(actual = report.microF1StandardDeviation() >= 0.0)
    }

    @Test
    internal fun `rejects an empty corpus`() {
        assertFailsWith<IllegalArgumentException> {
            validator.crossValidate(
                corpus = emptyList(),
                vectorizerFactory = { vectorizer() },
                learnerFactory = { learner() },
            )
        }
    }

    /**
     * Counts how a fold is featurised. Cross-validation runs every row through a vectorizer once
     * per fold, so a per-row loop here is one inference call — or one HTTP round trip — per row
     * per fold.
     */
    private class CountingBatchVectorizer(private val width: Int) : BatchVectorizer {
        var batchCalls: Int = 0
            private set

        var singleCalls: Int = 0
            private set

        override fun vectorize(text: String): FeatureVector {
            singleCalls += 1
            return one(text = text)
        }

        override fun vectorizeAll(texts: List<String>): List<FeatureVector> {
            batchCalls += 1
            return texts.map { text -> one(text = text) }
        }

        override fun dimension(): Int = width

        override fun fingerprint(): VectorizerFingerprint =
            VectorizerFingerprint(kind = "counting", dimension = width, configDigest = "d")

        private fun one(text: String): FeatureVector {
            val bucket = Math.floorMod(text.hashCode(), width)
            return FeatureVector(indices = intArrayOf(bucket), values = floatArrayOf(1.0f))
        }
    }

    @Test
    internal fun `each fold featurises in batches rather than row by row`() {
        val counting = CountingBatchVectorizer(width = featureCount)

        validator.crossValidate(
            corpus = ruleGeneratedCorpus(grouped = true),
            vectorizerFactory = { counting },
            learnerFactory = { learner() },
            folds = 5,
        )

        // Two calls per fold: the training rows, then the holdout rows.
        assertEquals(expected = 10, actual = counting.batchCalls)
        assertEquals(
            expected = 0,
            actual = counting.singleCalls,
            message = "cross-validation fell back to per-row vectorization despite a BatchVectorizer",
        )
    }
}
