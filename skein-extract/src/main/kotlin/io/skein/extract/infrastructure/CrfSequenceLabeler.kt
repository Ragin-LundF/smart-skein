package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import io.skein.extract.spi.SequenceLabeler
import io.skein.text.domain.Token
import kotlin.math.exp
import kotlin.math.ln

/**
 * A linear-chain Conditional Random Field for token tagging.
 *
 * The model scores a tag sequence by per-token **state features** and **transition features**
 * (previous-tag → tag, plus a start transition). State features include the token type, its
 * lowercased text, character prefix/suffix affixes, and the neighbouring tokens' types. Decoding
 * uses Viterbi; training does online SGD on the conditional log-likelihood, whose gradient is the
 * difference between gold feature counts and the model's expected counts — the latter obtained from
 * a log-space forward-backward pass. The step size follows a `1 / (1 + decayRate * step)` schedule
 * ([decayRate] = 0 keeps it constant). Tags are discovered from the training data.
 */
class CrfSequenceLabeler(
    private val initialLearningRate: Double = DEFAULT_LEARNING_RATE,
    private val decayRate: Double = DEFAULT_DECAY_RATE,
    private val l2Regularization: Double = DEFAULT_L2_REGULARIZATION,
) : SequenceLabeler {

    private val tagOrder = ArrayList<Tag>()
    private val knownTags = HashSet<Tag>()
    private val stateWeights = HashMap<Pair<Tag, String>, Double>()
    private val transitionWeights = HashMap<Pair<Tag, Tag>, Double>()
    private val startWeights = HashMap<Tag, Double>()
    private var step = 0L
    private var currentLearningRate = DEFAULT_LEARNING_RATE

    // How often each feature string was observed during training. Used only when saving a model, to
    // drop rare lexical features that are more likely to be personal data than vocabulary.
    // ponytail: one Int per distinct feature, never evicted, so it grows with the vocabulary.
    // Upgrade path: a count-min sketch, or counting only the `word=` features.
    private val featureCounts = HashMap<String, Int>()

    override fun learn(tokens: List<Token>, tags: List<Tag>) {
        require(value = tokens.size == tags.size) { "tokens and tags must align in length" }
        if (tokens.isEmpty()) {
            return
        }
        tags.forEach { tag -> registerTag(tag = tag) }
        currentLearningRate = initialLearningRate / (1.0 + decayRate * step)
        val features = extractFeatures(tokens = tokens)
        countFeatures(features = features)
        val alpha = forwardScores(features = features)
        val beta = backwardScores(features = features)
        val logZ = logSumExp(values = alpha.last())
        updateStateAndStart(features = features, tags = tags, alpha = alpha, beta = beta, logZ = logZ)
        updateTransitions(features = features, tags = tags, alpha = alpha, beta = beta, logZ = logZ)
        step++
    }

    override fun label(tokens: List<Token>): List<Tag> {
        check(value = tagOrder.isNotEmpty()) { "labeler has not been trained" }
        if (tokens.isEmpty()) {
            return emptyList()
        }
        return viterbi(features = extractFeatures(tokens = tokens))
    }

    /** Tallies feature occurrences for the save-time retention filter. Never affects scoring. */
    private fun countFeatures(features: List<List<String>>) {
        features.forEach { perToken ->
            perToken.forEach { feature ->
                featureCounts[feature] = (featureCounts[feature] ?: 0) + 1
            }
        }
    }

    private fun registerTag(tag: Tag) {
        if (knownTags.add(tag)) {
            tagOrder.add(tag)
        }
    }

    private fun extractFeatures(tokens: List<Token>): List<List<String>> {
        return tokens.indices.map { position -> featuresAt(tokens = tokens, position = position) }
    }

    private fun featuresAt(tokens: List<Token>, position: Int): List<String> {
        val token = tokens[position]
        val text = token.text.lowercase()
        val previousType = if (position > 0) tokens[position - 1].type.name else BOUNDARY
        val nextType = if (position < tokens.lastIndex) tokens[position + 1].type.name else BOUNDARY
        return listOf(
            "$TYPE_PREFIX${token.type.name}",
            "$WORD_PREFIX$text",
            "$PREFIX_PREFIX${text.take(n = AFFIX_LENGTH)}",
            "$SUFFIX_PREFIX${text.takeLast(n = AFFIX_LENGTH)}",
            "$PREV_TYPE_PREFIX$previousType",
            "$NEXT_TYPE_PREFIX$nextType",
        )
    }

    private fun stateScore(tagIndex: Int, features: List<String>): Double {
        val tag = tagOrder[tagIndex]
        return features.sumOf { feature -> stateWeights[tag to feature] ?: 0.0 }
    }

    private fun transitionScore(fromIndex: Int, toIndex: Int): Double {
        return transitionWeights[tagOrder[fromIndex] to tagOrder[toIndex]] ?: 0.0
    }

    private fun startScore(tagIndex: Int): Double {
        return startWeights[tagOrder[tagIndex]] ?: 0.0
    }

    private fun forwardScores(features: List<List<String>>): Array<DoubleArray> {
        val length = features.size
        val tagCount = tagOrder.size
        val alpha = Array(length) { DoubleArray(tagCount) }
        for (tag in 0 until tagCount) {
            alpha[0][tag] = startScore(tagIndex = tag) + stateScore(tagIndex = tag, features = features[0])
        }
        for (position in 1 until length) {
            for (tag in 0 until tagCount) {
                val incoming = DoubleArray(tagCount) { previous ->
                    alpha[position - 1][previous] + transitionScore(fromIndex = previous, toIndex = tag)
                }
                alpha[position][tag] = logSumExp(values = incoming) +
                    stateScore(tagIndex = tag, features = features[position])
            }
        }
        return alpha
    }

    private fun backwardScores(features: List<List<String>>): Array<DoubleArray> {
        val length = features.size
        val tagCount = tagOrder.size
        val beta = Array(length) { DoubleArray(tagCount) }
        for (position in length - 2 downTo 0) {
            for (tag in 0 until tagCount) {
                val outgoing = DoubleArray(tagCount) { next ->
                    transitionScore(fromIndex = tag, toIndex = next) +
                        stateScore(tagIndex = next, features = features[position + 1]) + beta[position + 1][next]
                }
                beta[position][tag] = logSumExp(values = outgoing)
            }
        }
        return beta
    }

    private fun updateStateAndStart(
        features: List<List<String>>,
        tags: List<Tag>,
        alpha: Array<DoubleArray>,
        beta: Array<DoubleArray>,
        logZ: Double,
    ) {
        for (position in features.indices) {
            for (tagIndex in tagOrder.indices) {
                val marginal = exp(x = alpha[position][tagIndex] + beta[position][tagIndex] - logZ)
                val gold = if (tags[position] == tagOrder[tagIndex]) 1.0 else 0.0
                val delta = gold - marginal
                applyStateGradient(tag = tagOrder[tagIndex], features = features[position], delta = delta)
                if (position == 0) {
                    applyStartGradient(tag = tagOrder[tagIndex], delta = delta)
                }
            }
        }
    }

    private fun updateTransitions(
        features: List<List<String>>,
        tags: List<Tag>,
        alpha: Array<DoubleArray>,
        beta: Array<DoubleArray>,
        logZ: Double,
    ) {
        for (position in 1 until features.size) {
            for (from in tagOrder.indices) {
                for (to in tagOrder.indices) {
                    val edge = exp(
                        x = alpha[position - 1][from] + transitionScore(fromIndex = from, toIndex = to) +
                            stateScore(tagIndex = to, features = features[position]) + beta[position][to] - logZ,
                    )
                    val gold = if (tags[position - 1] == tagOrder[from] && tags[position] == tagOrder[to]) 1.0 else 0.0
                    applyTransitionGradient(from = tagOrder[from], to = tagOrder[to], delta = gold - edge)
                }
            }
        }
    }

    private fun applyStateGradient(tag: Tag, features: List<String>, delta: Double) {
        for (feature in features) {
            val key = tag to feature
            val current = stateWeights[key] ?: 0.0
            stateWeights[key] = current + currentLearningRate * (delta - l2Regularization * current)
        }
    }

    private fun applyStartGradient(tag: Tag, delta: Double) {
        val current = startWeights[tag] ?: 0.0
        startWeights[tag] = current + currentLearningRate * (delta - l2Regularization * current)
    }

    private fun applyTransitionGradient(from: Tag, to: Tag, delta: Double) {
        val key = from to to
        val current = transitionWeights[key] ?: 0.0
        transitionWeights[key] = current + currentLearningRate * (delta - l2Regularization * current)
    }

    private fun viterbi(features: List<List<String>>): List<Tag> {
        val length = features.size
        val tagCount = tagOrder.size
        val best = Array(length) { DoubleArray(tagCount) }
        val backPointer = Array(length) { IntArray(tagCount) }
        for (tag in 0 until tagCount) {
            best[0][tag] = startScore(tagIndex = tag) + stateScore(tagIndex = tag, features = features[0])
        }
        for (position in 1 until length) {
            for (tag in 0 until tagCount) {
                val previous = bestPrevious(previousScores = best[position - 1], tag = tag)
                backPointer[position][tag] = previous
                best[position][tag] = best[position - 1][previous] +
                    transitionScore(fromIndex = previous, toIndex = tag) +
                    stateScore(tagIndex = tag, features = features[position])
            }
        }
        return backtrack(best = best, backPointer = backPointer)
    }

    private fun bestPrevious(previousScores: DoubleArray, tag: Int): Int {
        var bestIndex = 0
        var bestValue = Double.NEGATIVE_INFINITY
        for (previous in previousScores.indices) {
            val value = previousScores[previous] + transitionScore(fromIndex = previous, toIndex = tag)
            if (value > bestValue) {
                bestValue = value
                bestIndex = previous
            }
        }
        return bestIndex
    }

    private fun backtrack(best: Array<DoubleArray>, backPointer: Array<IntArray>): List<Tag> {
        val length = best.size
        var current = indexOfMax(scores = best[length - 1])
        val tagIndices = IntArray(length)
        tagIndices[length - 1] = current
        for (position in length - 1 downTo 1) {
            current = backPointer[position][current]
            tagIndices[position - 1] = current
        }
        return tagIndices.map { index -> tagOrder[index] }
    }

    /**
     * Captures everything needed to restore this labeler, as defensive copies — a snapshot taken
     * while training continues is not mutated by later [learn] calls.
     *
     * `currentLearningRate` is deliberately absent: it is recomputed from the initial rate, the
     * decay rate and [step] on every [learn], so persisting those three restores it exactly, and
     * storing it as well would create a second source of truth that could contradict them.
     */
    fun snapshot(): CrfModelSnapshot {
        return CrfModelSnapshot(
            tagOrder = tagOrder.toList(),
            startWeights = startWeights.toMap(),
            transitionWeights = transitionWeights.toMap(),
            stateWeights = stateWeights.toMap(),
            featureCounts = featureCounts.toMap(),
            initialLearningRate = initialLearningRate,
            decayRate = decayRate,
            l2Regularization = l2Regularization,
            step = step,
        )
    }

    companion object {

        /**
         * Rebuilds a labeler from [snapshot], including its hyperparameters and training progress,
         * so training can continue exactly where it stopped.
         *
         * A factory rather than an instance `restore`: restoring into a labeler constructed with
         * different hyperparameters would produce an object whose constructor arguments contradict
         * its own state.
         */
        fun from(snapshot: CrfModelSnapshot): CrfSequenceLabeler {
            require(value = snapshot.tagOrder.isNotEmpty()) { "a snapshot must carry at least one tag" }
            require(value = snapshot.tagOrder.toSet().size == snapshot.tagOrder.size) {
                "a snapshot must not repeat a tag"
            }
            require(value = snapshot.step >= 0L) { "step must not be negative" }
            val known = snapshot.tagOrder.toSet()
            snapshot.startWeights.keys.forEach { tag ->
                require(value = tag in known) { "start weight references unknown tag '${tag.value}'" }
            }
            snapshot.transitionWeights.keys.forEach { (from, to) ->
                require(value = from in known && to in known) { "transition references an unknown tag" }
            }
            snapshot.stateWeights.keys.forEach { (tag, _) ->
                require(value = tag in known) { "state weight references unknown tag '${tag.value}'" }
            }

            val labeler = CrfSequenceLabeler(
                initialLearningRate = snapshot.initialLearningRate,
                decayRate = snapshot.decayRate,
                l2Regularization = snapshot.l2Regularization,
            )
            snapshot.tagOrder.forEach { tag -> labeler.registerTag(tag = tag) }
            labeler.startWeights.putAll(snapshot.startWeights)
            labeler.transitionWeights.putAll(snapshot.transitionWeights)
            labeler.stateWeights.putAll(snapshot.stateWeights)
            labeler.featureCounts.putAll(snapshot.featureCounts)
            labeler.step = snapshot.step
            return labeler
        }

        /** Index of the largest score, ties resolved toward the lowest index. */
        private fun indexOfMax(scores: DoubleArray): Int {
            var bestIndex = 0
            for (index in scores.indices) {
                if (scores[index] > scores[bestIndex]) {
                    bestIndex = index
                }
            }
            return bestIndex
        }

        /** Numerically stable `ln(sum(exp(values)))`. */
        private fun logSumExp(values: DoubleArray): Double {
            val max = values.max()
            if (max == Double.NEGATIVE_INFINITY) {
                return max
            }
            var sum = 0.0
            for (value in values) {
                sum += exp(x = value - max)
            }
            return max + ln(x = sum)
        }

        private const val DEFAULT_LEARNING_RATE = 0.1
        private const val DEFAULT_DECAY_RATE = 0.0
        private const val DEFAULT_L2_REGULARIZATION = 0.0
        private const val AFFIX_LENGTH = 3
        private const val BOUNDARY = "^"

        internal const val TYPE_PREFIX = "type="
        internal const val WORD_PREFIX = "word="
        internal const val PREFIX_PREFIX = "prefix="
        internal const val SUFFIX_PREFIX = "suffix="
        internal const val PREV_TYPE_PREFIX = "prevType="
        internal const val NEXT_TYPE_PREFIX = "nextType="

        /**
         * Feature prefixes whose values are raw training text. The single source of truth for the
         * save-time privacy filter — adding a lexical feature without listing it here would let that
         * text reach disk in a mode that promises otherwise.
         */
        internal val LEXICAL_FEATURE_PREFIXES = listOf(WORD_PREFIX, PREFIX_PREFIX, SUFFIX_PREFIX)
    }
}
