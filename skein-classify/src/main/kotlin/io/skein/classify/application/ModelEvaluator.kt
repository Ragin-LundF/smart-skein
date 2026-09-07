package io.skein.classify.application

import io.skein.classify.domain.CrossValidationReport
import io.skein.classify.domain.DatasetSplit
import io.skein.classify.domain.EvaluationReport
import io.skein.classify.domain.EvaluationReportFactory
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.domain.PredictionOutcome
import io.skein.classify.domain.Record
import io.skein.classify.spi.Classifier
import kotlin.random.Random

/** Rank depth reported as top-k accuracy unless overridden. */
private const val DEFAULT_TOP_K = 3

/** Number of reliability buckets unless overridden. */
private const val DEFAULT_CALIBRATION_BINS = 10

/** Fraction held out by [ModelEvaluator.holdout] unless overridden. */
private const val DEFAULT_TEST_RATIO = 0.2

/** Folds used by [ModelEvaluator.crossValidate] unless overridden. */
private const val DEFAULT_FOLDS = 5

/** Training passes over a split unless overridden. */
private const val DEFAULT_EPOCHS = 1

/**
 * Measures model quality: scores a trained classifier against held-out data, or trains fresh models
 * over a split or k folds of a corpus and scores those.
 *
 * **What is being measured.** [evaluate] and [evaluateRecords] score the model you hand them, so
 * they measure that model — provided the data is genuinely held out. [holdout] and [crossValidate]
 * train *new* models from [classifierFactory], so they measure the **recipe** (classifier, corpus
 * and hyperparameters), not any particular trained artifact. In particular, running them over the
 * observations stored in a `.skein` file says nothing about the model saved in that same file,
 * because that model was trained on all of those rows.
 */
class ModelEvaluator(
    private val topK: Int = DEFAULT_TOP_K,
    private val calibrationBins: Int = DEFAULT_CALIBRATION_BINS,
    private val splitter: StratifiedSplitter = StratifiedSplitter(),
) {

    /**
     * Scores an already-trained [classifier] against [holdout]. Trains nothing and mutates nothing,
     * so the same classifier can be scored against several holdouts.
     */
    fun evaluate(classifier: Classifier, holdout: List<LabeledFeatures>): EvaluationReport {
        require(value = holdout.isNotEmpty()) { "cannot evaluate against an empty holdout" }
        return EvaluationReportFactory.from(
            outcomes = outcomesFor(classifier = classifier, holdout = holdout),
            topK = topK,
            calibrationBins = calibrationBins,
        )
    }

    /**
     * Scores a trained [service] against labeled [records], reading the ground truth from the
     * schema's label field. [records] must not have been learned from.
     */
    fun evaluateRecords(service: ClassificationService, records: List<Record>): EvaluationReport {
        require(value = records.isNotEmpty()) { "cannot evaluate an empty record list" }
        val mapper = RecordMapper(schema = service.schema)
        val outcomes = records.map { record ->
            val label = requireNotNull(value = mapper.map(record = record).label) {
                "record is missing a value for the label field '${service.schema.labelField.name}'"
            }
            PredictionOutcome(expected = label, prediction = service.classify(record = record))
        }
        return EvaluationReportFactory.from(
            outcomes = outcomes,
            topK = topK,
            calibrationBins = calibrationBins,
        )
    }

    /**
     * Stratified holdout: trains a fresh classifier from [classifierFactory] on the training split
     * and scores it against the holdout.
     */
    fun holdout(
        observations: List<LabeledFeatures>,
        classifierFactory: () -> Classifier,
        testRatio: Double = DEFAULT_TEST_RATIO,
        epochs: Int = DEFAULT_EPOCHS,
    ): EvaluationReport {
        val split = splitter.holdout(observations = observations, testRatio = testRatio)
        val classifier = trainFold(split = split, classifierFactory = classifierFactory, epochs = epochs, fold = 0)
        return evaluate(classifier = classifier, holdout = split.holdout)
    }

    /**
     * Stratified k-fold cross-validation. Calls [classifierFactory] exactly [folds] times — one
     * fresh model per fold — so folds cannot leak state into one another.
     */
    fun crossValidate(
        observations: List<LabeledFeatures>,
        classifierFactory: () -> Classifier,
        folds: Int = DEFAULT_FOLDS,
        epochs: Int = DEFAULT_EPOCHS,
    ): CrossValidationReport {
        val splits = splitter.folds(observations = observations, folds = folds)
        val pooledOutcomes = ArrayList<PredictionOutcome>(observations.size)
        val foldReports = splits.mapIndexed { index, split ->
            val classifier = trainFold(
                split = split,
                classifierFactory = classifierFactory,
                epochs = epochs,
                fold = index,
            )
            pooledOutcomes.addAll(elements = outcomesFor(classifier = classifier, holdout = split.holdout))
            evaluate(classifier = classifier, holdout = split.holdout)
        }
        return CrossValidationReport(
            folds = foldReports,
            pooled = EvaluationReportFactory.from(
                outcomes = pooledOutcomes,
                topK = topK,
                calibrationBins = calibrationBins,
            ),
        )
    }

    /**
     * Builds and trains one model over [DatasetSplit.training].
     *
     * ponytail: [epochs] above 1 only affects incremental classifiers. `NaiveBayesClassifier`
     * rebuilds its snapshot from the batch, so extra passes are a no-op; the SGD classifier
     * continues from its current weights, so passes accumulate. This class only sees the
     * [Classifier] port and cannot tell which it holds. Upgrade path: a `supportsEpochs` flag on
     * the port if the divergence ever bites.
     *
     * Fold isolation comes from [classifierFactory], never from `forget()` — relying on `forget()`
     * would silently bind correctness to two implementations' internal reset behaviour.
     */
    private fun trainFold(
        split: DatasetSplit,
        classifierFactory: () -> Classifier,
        epochs: Int,
        fold: Int,
    ): Classifier {
        require(value = epochs >= 1) { "epochs must be at least 1" }
        val classifier = classifierFactory()
        repeat(times = epochs) { epoch ->
            val ordered = if (epoch == 0) {
                split.training
            } else {
                split.training.shuffled(random = Random(seed = fold.toLong() + epoch))
            }
            classifier.learnAll(observations = ordered)
        }
        return classifier
    }

    private fun outcomesFor(classifier: Classifier, holdout: List<LabeledFeatures>): List<PredictionOutcome> {
        return holdout.map { observation ->
            PredictionOutcome(
                expected = observation.label,
                prediction = classifier.classify(features = observation.features),
            )
        }
    }
}
