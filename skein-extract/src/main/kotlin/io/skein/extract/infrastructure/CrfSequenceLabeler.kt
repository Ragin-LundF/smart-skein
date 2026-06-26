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

    override fun learn(tokens: List<Token>, tags: List<Tag>) {
        require(tokens.size == tags.size) { "tokens and tags must align in length" }
        if (tokens.isEmpty()) {
            return
        }
        tags.forEach { tag -> registerTag(tag) }
        currentLearningRate = initialLearningRate / (1.0 + decayRate * step)
        val features = extractFeatures(tokens)
        val alpha = forwardScores(features)
        val beta = backwardScores(features)
        val logZ = logSumExp(alpha.last())
        updateStateAndStart(features, tags, alpha, beta, logZ)
        updateTransitions(features, tags, alpha, beta, logZ)
        step++
    }

    override fun label(tokens: List<Token>): List<Tag> {
        check(tagOrder.isNotEmpty()) { "labeler has not been trained" }
        if (tokens.isEmpty()) {
            return emptyList()
        }
        return viterbi(extractFeatures(tokens))
    }

    private fun registerTag(tag: Tag) {
        if (knownTags.add(tag)) {
            tagOrder.add(tag)
        }
    }

    private fun extractFeatures(tokens: List<Token>): List<List<String>> {
        return tokens.indices.map { position -> featuresAt(tokens, position) }
    }

    private fun featuresAt(tokens: List<Token>, position: Int): List<String> {
        val token = tokens[position]
        val text = token.text.lowercase()
        val previousType = if (position > 0) tokens[position - 1].type.name else BOUNDARY
        val nextType = if (position < tokens.lastIndex) tokens[position + 1].type.name else BOUNDARY
        return listOf(
            "type=${token.type.name}",
            "word=$text",
            "prefix=${text.take(AFFIX_LENGTH)}",
            "suffix=${text.takeLast(AFFIX_LENGTH)}",
            "prevType=$previousType",
            "nextType=$nextType",
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
            alpha[0][tag] = startScore(tag) + stateScore(tag, features[0])
        }
        for (position in 1 until length) {
            for (tag in 0 until tagCount) {
                val incoming = DoubleArray(tagCount) { previous ->
                    alpha[position - 1][previous] + transitionScore(previous, tag)
                }
                alpha[position][tag] = logSumExp(incoming) + stateScore(tag, features[position])
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
                    transitionScore(tag, next) + stateScore(next, features[position + 1]) + beta[position + 1][next]
                }
                beta[position][tag] = logSumExp(outgoing)
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
                val marginal = exp(alpha[position][tagIndex] + beta[position][tagIndex] - logZ)
                val gold = if (tags[position] == tagOrder[tagIndex]) 1.0 else 0.0
                val delta = gold - marginal
                applyStateGradient(tagOrder[tagIndex], features[position], delta)
                if (position == 0) {
                    applyStartGradient(tagOrder[tagIndex], delta)
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
                        alpha[position - 1][from] + transitionScore(from, to) +
                            stateScore(to, features[position]) + beta[position][to] - logZ,
                    )
                    val gold = if (tags[position - 1] == tagOrder[from] && tags[position] == tagOrder[to]) 1.0 else 0.0
                    applyTransitionGradient(tagOrder[from], tagOrder[to], gold - edge)
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
            best[0][tag] = startScore(tag) + stateScore(tag, features[0])
        }
        for (position in 1 until length) {
            for (tag in 0 until tagCount) {
                val previous = bestPrevious(best[position - 1], tag)
                backPointer[position][tag] = previous
                best[position][tag] = best[position - 1][previous] + transitionScore(previous, tag) +
                    stateScore(tag, features[position])
            }
        }
        return backtrack(best, backPointer)
    }

    private fun bestPrevious(previousScores: DoubleArray, tag: Int): Int {
        var bestIndex = 0
        var bestValue = Double.NEGATIVE_INFINITY
        for (previous in previousScores.indices) {
            val value = previousScores[previous] + transitionScore(previous, tag)
            if (value > bestValue) {
                bestValue = value
                bestIndex = previous
            }
        }
        return bestIndex
    }

    private fun backtrack(best: Array<DoubleArray>, backPointer: Array<IntArray>): List<Tag> {
        val length = best.size
        var current = indexOfMax(best[length - 1])
        val tagIndices = IntArray(length)
        tagIndices[length - 1] = current
        for (position in length - 1 downTo 1) {
            current = backPointer[position][current]
            tagIndices[position - 1] = current
        }
        return tagIndices.map { index -> tagOrder[index] }
    }

    private fun indexOfMax(scores: DoubleArray): Int {
        var bestIndex = 0
        for (index in scores.indices) {
            if (scores[index] > scores[bestIndex]) {
                bestIndex = index
            }
        }
        return bestIndex
    }

    private fun logSumExp(values: DoubleArray): Double {
        val max = values.max()
        if (max == Double.NEGATIVE_INFINITY) {
            return max
        }
        var sum = 0.0
        for (value in values) {
            sum += exp(value - max)
        }
        return max + ln(sum)
    }

    private companion object {
        const val DEFAULT_LEARNING_RATE = 0.1
        const val DEFAULT_DECAY_RATE = 0.0
        const val DEFAULT_L2_REGULARIZATION = 0.0
        const val AFFIX_LENGTH = 3
        const val BOUNDARY = "^"
    }
}
