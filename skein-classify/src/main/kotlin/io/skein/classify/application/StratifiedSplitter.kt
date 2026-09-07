package io.skein.classify.application

import io.skein.classify.domain.DatasetSplit
import io.skein.classify.domain.LabeledFeatures
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.random.Random

/** Seed used when the caller does not supply one, so a split is reproducible by default. */
private const val DEFAULT_SEED = 42L

/** Fraction of the corpus held out by [StratifiedSplitter.holdout] unless overridden. */
private const val DEFAULT_TEST_RATIO = 0.2

/** Cross-validation needs at least a train and a test fold. */
private const val MIN_FOLDS = 2

/**
 * Partitions a labeled corpus while preserving each label's proportion, deterministically under
 * [seed].
 *
 * Label groups are visited in sorted label order rather than map-encounter order, so the same
 * corpus splits identically even when its rows arrive in a different sequence — which matters
 * because observation order in a `.skein` file follows ingestion.
 *
 * ponytail: materializes full lists per fold, so a k-fold run holds roughly `folds` copies of the
 * observation references. Upgrade path for corpora that do not fit in memory: index-based folds
 * over a [io.skein.classify.spi.FeatureStore] cursor.
 */
class StratifiedSplitter(private val seed: Long = DEFAULT_SEED) {

    /**
     * Holds out [testRatio] of each label's observations.
     *
     * Every label keeps at least one training example: without that clamp a rare class can vanish
     * from training entirely, the model can never predict it, and its recall is a structurally
     * guaranteed zero — an artifact of the split rather than a measurement of the model. A label
     * with a single observation therefore contributes nothing to the holdout.
     */
    fun holdout(observations: List<LabeledFeatures>, testRatio: Double = DEFAULT_TEST_RATIO): DatasetSplit {
        require(value = observations.isNotEmpty()) { "cannot split an empty corpus" }
        require(value = testRatio > 0.0 && testRatio < 1.0) { "testRatio must be between 0 and 1 exclusive" }

        val random = Random(seed = seed)
        val holdout = ArrayList<LabeledFeatures>()
        val training = ArrayList<LabeledFeatures>()
        for (group in sortedGroups(observations = observations)) {
            val shuffled = group.shuffled(random = random)
            val testCount = min(a = (shuffled.size * testRatio).roundToInt(), b = shuffled.size - 1)
            holdout.addAll(elements = shuffled.take(n = testCount))
            training.addAll(elements = shuffled.drop(n = testCount))
        }
        return DatasetSplit(training = training.shuffled(random = random), holdout = holdout)
    }

    /**
     * Splits into [folds] stratified folds. Each observation is held out by exactly one fold, and
     * fold sizes differ by at most one.
     *
     * A label with fewer members than [folds] is absent from some folds' training sets and will
     * score zero recall there. That is a property of the data, and surfacing it is the point.
     */
    fun folds(observations: List<LabeledFeatures>, folds: Int): List<DatasetSplit> {
        require(value = observations.isNotEmpty()) { "cannot split an empty corpus" }
        require(value = folds >= MIN_FOLDS) { "folds must be at least $MIN_FOLDS" }
        require(value = folds <= observations.size) { "folds must not exceed the number of observations" }

        val random = Random(seed = seed)
        val assignments = List(size = folds) { ArrayList<LabeledFeatures>() }
        // The round-robin continues across label groups instead of restarting at fold 0 for each.
        // Restarting would stack every label's remainder onto the low-numbered folds, so with many
        // labels fold 0 ends up markedly larger than the last fold.
        var offset = 0
        for (group in sortedGroups(observations = observations)) {
            // The group is shuffled first, so this round-robin is unbiased while still keeping each
            // label's proportion identical across folds to within one observation.
            group.shuffled(random = random).forEachIndexed { position, observation ->
                assignments[(offset + position) % folds].add(element = observation)
            }
            offset = (offset + group.size) % folds
        }
        return assignments.mapIndexed { index, foldHoldout ->
            val training = assignments
                .filterIndexed { other, _ -> other != index }
                .flatten()
            DatasetSplit(
                // A per-fold seed keeps folds uncorrelated while staying deterministic, mirroring
                // the `seed + epoch` idiom in ClassificationService.retrain.
                training = training.shuffled(random = Random(seed = seed + index)),
                holdout = foldHoldout,
            )
        }
    }

    private fun sortedGroups(observations: List<LabeledFeatures>): List<List<LabeledFeatures>> {
        return observations
            .groupBy { observation -> observation.label }
            .entries
            .sortedBy { entry -> entry.key.value }
            .map { entry -> entry.value }
    }
}
