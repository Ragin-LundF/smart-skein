package io.skein.classify.domain

/**
 * Builds [MultiLabelMetrics] from evaluated cases. Pure computation, shared by every caller that
 * scores multi-label output so one averaging convention is applied everywhere.
 *
 * Every ratio returns `0.0` rather than `NaN` when its denominator is zero, so a report over a
 * label the model never predicted is still printable and averageable.
 */
object MultiLabelMetricsFactory {

    /**
     * Scores [outcomes] at the threshold each prediction already carries.
     *
     * The label universe is every label that was expected or predicted anywhere in [outcomes], so
     * a label the model invented is counted against it, and a label that never occurred at all is
     * silently absent rather than contributing a spurious zero.
     */
    fun from(outcomes: List<MultiLabelOutcome>): MultiLabelMetrics {
        require(value = outcomes.isNotEmpty()) { "cannot build metrics from an empty outcome list" }
        val threshold = outcomes.first().prediction.threshold
        val predictedSets = outcomes.map { outcome -> outcome.prediction.labels() }
        val perLabel = perLabelMetrics(outcomes = outcomes, predictedSets = predictedSets)
        return MultiLabelMetrics(
            sampleCount = outcomes.size,
            threshold = threshold,
            micro = microAverage(outcomes = outcomes, predictedSets = predictedSets),
            macro = macroAverage(perLabel = perLabel),
            exactMatchRatio = ratio(
                numerator = outcomes.indices.count { i -> outcomes[i].expected == predictedSets[i] },
                denominator = outcomes.size,
            ),
            coverage = ratio(
                numerator = predictedSets.count { predicted -> predicted.isNotEmpty() },
                denominator = outcomes.size,
            ),
            perLabel = perLabel,
        )
    }

    /**
     * The fraction of expected labels that appear among each case's [count] highest-scoring labels,
     * ignoring the threshold entirely.
     *
     * This is the number that says whether a human reviewer shown [count] candidates would find the
     * right answer. Cases with no expected label contribute nothing, since there is nothing to
     * recall; when no case has an expected label the result is `0.0`.
     */
    fun recallAtK(outcomes: List<MultiLabelOutcome>, count: Int): Double {
        require(value = count > 0) { "count must be positive, got $count" }
        var found = 0
        var expected = 0
        outcomes.forEach { outcome ->
            val candidates = outcome.prediction.topK(count = count).map { scored -> scored.label }.toSet()
            found += outcome.expected.count { label -> label in candidates }
            expected += outcome.expected.size
        }
        return ratio(numerator = found, denominator = expected)
    }

    private fun microAverage(
        outcomes: List<MultiLabelOutcome>,
        predictedSets: List<Set<Label>>,
    ): AveragedMetrics {
        var truePositives = 0
        var falsePositives = 0
        var falseNegatives = 0
        outcomes.indices.forEach { i ->
            val expected = outcomes[i].expected
            val predicted = predictedSets[i]
            truePositives += predicted.count { label -> label in expected }
            falsePositives += predicted.count { label -> label !in expected }
            falseNegatives += expected.count { label -> label !in predicted }
        }
        return averaged(
            truePositives = truePositives,
            falsePositives = falsePositives,
            falseNegatives = falseNegatives,
        )
    }

    private fun macroAverage(perLabel: List<LabelMetrics>): AveragedMetrics {
        if (perLabel.isEmpty()) {
            return AveragedMetrics(precision = 0.0, recall = 0.0, f1 = 0.0)
        }
        return AveragedMetrics(
            precision = perLabel.sumOf { metrics -> metrics.precision } / perLabel.size,
            recall = perLabel.sumOf { metrics -> metrics.recall } / perLabel.size,
            f1 = perLabel.sumOf { metrics -> metrics.f1 } / perLabel.size,
        )
    }

    private fun perLabelMetrics(
        outcomes: List<MultiLabelOutcome>,
        predictedSets: List<Set<Label>>,
    ): List<LabelMetrics> {
        val universe = sortedSetOf<String>()
        outcomes.forEach { outcome -> outcome.expected.forEach { label -> universe.add(element = label.value) } }
        predictedSets.forEach { predicted -> predicted.forEach { label -> universe.add(element = label.value) } }
        return universe.map { value ->
            val label = Label(value = value)
            var truePositives = 0
            var falsePositives = 0
            var falseNegatives = 0
            outcomes.indices.forEach { i ->
                val isExpected = label in outcomes[i].expected
                val isPredicted = label in predictedSets[i]
                when {
                    isExpected && isPredicted -> truePositives++
                    isPredicted -> falsePositives++
                    isExpected -> falseNegatives++
                }
            }
            val averages = averaged(
                truePositives = truePositives,
                falsePositives = falsePositives,
                falseNegatives = falseNegatives,
            )
            LabelMetrics(
                label = label,
                precision = averages.precision,
                recall = averages.recall,
                f1 = averages.f1,
                support = truePositives + falseNegatives,
            )
        }
    }

    private fun averaged(truePositives: Int, falsePositives: Int, falseNegatives: Int): AveragedMetrics {
        val precision = ratio(numerator = truePositives, denominator = truePositives + falsePositives)
        val recall = ratio(numerator = truePositives, denominator = truePositives + falseNegatives)
        val denominator = precision + recall
        val f1 = if (denominator == 0.0) 0.0 else 2.0 * precision * recall / denominator
        return AveragedMetrics(precision = precision, recall = recall, f1 = f1)
    }

    private fun ratio(numerator: Int, denominator: Int): Double {
        return if (denominator == 0) 0.0 else numerator.toDouble() / denominator
    }
}
