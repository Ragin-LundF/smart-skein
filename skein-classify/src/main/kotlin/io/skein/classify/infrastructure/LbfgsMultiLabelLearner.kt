package io.skein.classify.infrastructure

import io.skein.classify.domain.ClassifierHyperparameters
import io.skein.classify.domain.Label
import io.skein.classify.domain.MultiLabeledFeatures
import io.skein.classify.domain.SparseMatrix
import io.skein.classify.domain.SparseMatrixBuilder
import io.skein.classify.spi.BatchLearner
import io.skein.classify.spi.MultiLabelClassifier
import java.util.concurrent.Callable
import java.util.concurrent.Executors

/** Inverse regularisation strength, on scikit-learn's scale: larger means less regularisation. */
private const val DEFAULT_INVERSE_REGULARIZATION = 10.0

/** Convergence test on the largest absolute gradient component. */
private const val DEFAULT_GRADIENT_TOLERANCE = 1e-4

/** Iterations one label may take before the fit reports it unconverged and moves on. */
private const val DEFAULT_MAX_ITERATIONS = 500

/** Curvature pairs each worker retains. Five is the memory-conscious end of the usual 3-20 range. */
private const val DEFAULT_HISTORY_SIZE = 5

/** Share of weights kept by magnitude. Measured: 8% changed one label decision in 1.7 million. */
private const val DEFAULT_KEEP_FRACTION = 0.08

/**
 * Fits an independent L2-regularised logistic head per label with L-BFGS, all over one shared
 * design matrix.
 *
 * **One-vs-rest, not softmax.** Each label's head is fitted against "this label, or not this label",
 * so nothing constrains the heads to agree and a record can come back with several labels or none.
 * The labels share the design matrix and nothing else, which is also what makes fitting them
 * embarrassingly parallel.
 *
 * Parallelism is bounded by [TrainingParallelism] from the heap and the feature count, not from the
 * core count. See that type for why the obvious `parallelStream()` is a memory trap at wide feature
 * spaces.
 *
 * [featureCount] must be the vectorizer's full feature space, not the widest index this corpus
 * happens to use. Deriving it from the data would give each cross-validation fold a different
 * feature space, and a model fitted on one fold could not score a record from another.
 */
class LbfgsMultiLabelLearner(
    private val featureCount: Int,
    private val inverseRegularization: Double = DEFAULT_INVERSE_REGULARIZATION,
    private val gradientTolerance: Double = DEFAULT_GRADIENT_TOLERANCE,
    private val maxIterations: Int = DEFAULT_MAX_ITERATIONS,
    private val historySize: Int = DEFAULT_HISTORY_SIZE,
    private val keepFraction: Double = DEFAULT_KEEP_FRACTION,
) : BatchLearner {

    init {
        require(value = featureCount > 0) { "featureCount must be positive, got $featureCount" }
        require(value = inverseRegularization > 0.0) { "inverseRegularization must be positive" }
        require(value = keepFraction > 0.0 && keepFraction <= 1.0) {
            "keepFraction must be in (0, 1], got $keepFraction"
        }
    }

    override fun fit(observations: List<MultiLabeledFeatures>): MultiLabelClassifier {
        require(value = observations.isNotEmpty()) { "cannot fit a model on an empty corpus" }
        val labels = observations.flatMap { observation -> observation.labels }
            .map { label -> label.value }
            .distinct()
            .sorted()
            .map { value -> Label(value = value) }
        require(value = labels.isNotEmpty()) {
            "cannot fit a model on a corpus where no observation carries a label"
        }
        val matrix = designMatrix(observations = observations)
        val positiveRows = positiveRowsByLabel(observations = observations, labels = labels)
        val fitted = fitAll(matrix = matrix, labels = labels, positiveRows = positiveRows)

        val intercepts = DoubleArray(size = labels.size) { index -> fitted[index][featureCount].toDouble() }
        val weightsByLabel = fitted.map { parameters -> parameters.copyOf(newSize = featureCount) }
        return MultiLabelLogisticClassifier(
            weights = MultiLabelWeights.fromLabelMajor(
                labels = labels,
                intercepts = intercepts,
                weightsByLabel = weightsByLabel,
                featureCount = featureCount,
                keepFraction = keepFraction,
            ),
            tuning = ClassifierHyperparameters(l2Regularization = 1.0 / inverseRegularization),
        )
    }

    /**
     * Fits every label, at most [TrainingParallelism.workerCount] at a time.
     *
     * Results are collected as `FloatArray` rather than `DoubleArray`: the fit needs double
     * precision, the stored weight does not, and holding 237 dense double vectors over 100,000
     * features at once is 194 MB against 97 MB for no accuracy that survives pruning anyway.
     */
    private fun fitAll(
        matrix: SparseMatrix,
        labels: List<Label>,
        positiveRows: List<IntArray>,
    ): List<FloatArray> {
        val workers = TrainingParallelism.workerCount(featureCount = featureCount, historySize = historySize)
        val pool = Executors.newFixedThreadPool(workers)
        return try {
            val tasks = labels.indices.map { index ->
                Callable { fitOne(matrix = matrix, positiveRows = positiveRows[index]) }
            }
            pool.invokeAll(tasks).map { future -> future.get() }
        } finally {
            pool.shutdown()
        }
    }

    private fun fitOne(matrix: SparseMatrix, positiveRows: IntArray): FloatArray {
        val objective = LogisticObjective(
            matrix = matrix,
            positiveRows = positiveRows,
            inverseRegularization = inverseRegularization,
        )
        val minimizer = LbfgsMinimizer(
            historySize = historySize,
            maxIterations = maxIterations,
            gradientTolerance = gradientTolerance,
        )
        val result = minimizer.minimize(function = objective)
        return FloatArray(size = result.point.size) { index -> result.point[index].toFloat() }
    }

    private fun designMatrix(observations: List<MultiLabeledFeatures>): SparseMatrix {
        val builder = SparseMatrixBuilder(columnCount = featureCount)
        observations.forEach { observation -> builder.addRow(vector = observation.features) }
        return builder.build()
    }

    /** For each label, the ascending row indices where it is present — what a one-vs-rest head needs. */
    private fun positiveRowsByLabel(
        observations: List<MultiLabeledFeatures>,
        labels: List<Label>,
    ): List<IntArray> {
        val rowsByLabel = labels.associateWith { ArrayList<Int>() }
        observations.forEachIndexed { row, observation ->
            observation.labels.forEach { label -> rowsByLabel.getValue(key = label).add(element = row) }
        }
        return labels.map { label -> rowsByLabel.getValue(key = label).toIntArray() }
    }
}
