package io.skein.classify.application

import io.skein.classify.domain.MultiLabelCrossValidationReport
import io.skein.classify.domain.MultiLabelMetricsFactory
import io.skein.classify.domain.MultiLabelOutcome
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.MultiLabeledText
import io.skein.classify.spi.BatchLearner
import io.skein.classify.spi.Vectorizer

/** Folds used unless the caller overrides them. */
private const val DEFAULT_FOLDS = 5

/** Acceptance threshold used unless the caller overrides it. */
private const val DEFAULT_THRESHOLD = 0.5

/**
 * Cross-validates a multi-label *recipe* — a vectorizer, a learner and a corpus — by fitting a
 * fresh model per fold.
 *
 * Read that as written: this measures the recipe, never a particular trained artifact. Running it
 * over the corpus a saved model was trained on says nothing about that model, because the model
 * already saw every row.
 *
 * **Featurisation happens inside the fold, not before it.** The corpus arrives as
 * [MultiLabeledText] and [crossValidate] builds a vectorizer per fold from that fold's training
 * rows alone. With plain feature hashing there is nothing fitted and nothing to leak, so this looks
 * like ceremony — but the moment a document-frequency table, an IDF weighting or a fitted
 * vocabulary enters the pipeline, fitting it once over the whole corpus quietly moves held-out
 * statistics into the training features and every score afterwards is wrong in the flattering
 * direction. The structure is here so that change stays a one-line substitution.
 *
 * Pair it with [GroupedSplitter] — which it uses by default — whenever labels came from a rule
 * engine or any other generator, and read that class for what a random split costs.
 */
class MultiLabelCrossValidator(
    private val splitter: GroupedSplitter = GroupedSplitter(),
    private val evaluator: MultiLabelEvaluator = MultiLabelEvaluator(),
) {

    /**
     * Fits and scores [folds] models over [corpus].
     *
     * [vectorizerFactory] is handed each fold's **training rows only** and must return a vectorizer
     * fitted to them. [learnerFactory] is called once per fold, so folds cannot share state.
     */
    fun crossValidate(
        corpus: List<MultiLabeledText>,
        vectorizerFactory: (List<MultiLabeledText>) -> Vectorizer,
        learnerFactory: () -> BatchLearner,
        folds: Int = DEFAULT_FOLDS,
        threshold: Double = DEFAULT_THRESHOLD,
    ): MultiLabelCrossValidationReport {
        require(value = corpus.isNotEmpty()) { "cannot cross-validate an empty corpus" }
        val assignment = splitter.assign(groups = corpus.map { row -> row.group }, folds = folds)
        val pooledOutcomes = ArrayList<MultiLabelOutcome>(corpus.size)
        val foldMetrics = (0 until folds).map { fold ->
            val training = corpus.filterIndexed { position, _ -> assignment[position] != fold }
            val holdout = corpus.filterIndexed { position, _ -> assignment[position] == fold }
            require(value = holdout.isNotEmpty()) { "fold $fold received no rows" }
            val outcomes = scoreFold(
                training = training,
                holdout = holdout,
                vectorizerFactory = vectorizerFactory,
                learnerFactory = learnerFactory,
                threshold = threshold,
            )
            pooledOutcomes.addAll(elements = outcomes)
            MultiLabelMetricsFactory.from(outcomes = outcomes)
        }
        return MultiLabelCrossValidationReport(
            folds = foldMetrics,
            pooled = MultiLabelMetricsFactory.from(outcomes = pooledOutcomes),
            pooledOutcomes = pooledOutcomes,
        )
    }

    private fun scoreFold(
        training: List<MultiLabeledText>,
        holdout: List<MultiLabeledText>,
        vectorizerFactory: (List<MultiLabeledText>) -> Vectorizer,
        learnerFactory: () -> BatchLearner,
        threshold: Double,
    ): List<MultiLabelOutcome> {
        val vectorizer = vectorizerFactory(training)
        val model = learnerFactory().fit(
            observations = training.map { row -> featurize(row = row, vectorizer = vectorizer) },
        )
        return evaluator.outcomes(
            classifier = model,
            holdout = holdout.map { row -> featurize(row = row, vectorizer = vectorizer) },
            threshold = threshold,
        )
    }

    private fun featurize(row: MultiLabeledText, vectorizer: Vectorizer): MultiLabeledFeatures {
        return MultiLabeledFeatures(
            features = vectorizer.vectorize(text = row.featureText),
            labels = row.labels,
            group = row.group,
        )
    }
}
