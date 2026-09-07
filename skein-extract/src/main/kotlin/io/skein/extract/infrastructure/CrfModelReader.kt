package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import java.io.DataInputStream
import java.time.Instant

/**
 * Reads the `SKCR` payload written by [CrfModelWriter].
 *
 * Every count and index is validated before it is used to allocate or to index, so a corrupt file
 * fails with a clear error rather than exhausting memory or throwing an obscure array exception.
 * The reader never asserts end-of-stream, which is what gives the format its forward compatibility.
 */
internal class CrfModelReader(private val input: DataInputStream) {

    fun readMetadata(formatMajor: Int, formatMinor: Int): CrfModelMetadata {
        val createdAt = input.readLong()
        val writerVersion = readString()
        val retention = CrfModelFormat.retentionOf(code = input.readInt())
        val minLexicalOccurrences = input.readInt()
        return CrfModelMetadata(
            formatMajor = formatMajor,
            formatMinor = formatMinor,
            createdAt = Instant.ofEpochMilli(createdAt),
            writerVersion = writerVersion,
            retention = retention,
            minLexicalOccurrences = minLexicalOccurrences,
        )
    }

    fun readSnapshot(): CrfModelSnapshot {
        val initialLearningRate = input.readDouble()
        val decayRate = input.readDouble()
        val l2Regularization = input.readDouble()
        val step = input.readLong()

        val tags = readTags()
        val startWeights = readStartWeights(tags = tags)
        val transitionWeights = readTransitions(tags = tags)
        val features = readFeatureTable()
        val stateWeights = readStateWeights(tags = tags, features = features.first)

        return CrfModelSnapshot(
            tagOrder = tags,
            startWeights = startWeights,
            transitionWeights = transitionWeights,
            stateWeights = stateWeights,
            featureCounts = features.second,
            initialLearningRate = initialLearningRate,
            decayRate = decayRate,
            l2Regularization = l2Regularization,
            step = step,
        )
    }

    private fun readTags(): List<Tag> {
        val count = readCount(what = "tag")
        return List(size = count) { Tag(value = readString()) }
    }

    private fun readStartWeights(tags: List<Tag>): Map<Tag, Double> {
        val count = readCount(what = "start weight")
        val weights = HashMap<Tag, Double>(count)
        repeat(times = count) {
            val tag = tags[readIndex(bound = tags.size, what = "tag")]
            weights[tag] = input.readDouble()
        }
        return weights
    }

    private fun readTransitions(tags: List<Tag>): Map<Pair<Tag, Tag>, Double> {
        val count = readCount(what = "transition")
        val weights = HashMap<Pair<Tag, Tag>, Double>(count)
        repeat(times = count) {
            val from = tags[readIndex(bound = tags.size, what = "tag")]
            val to = tags[readIndex(bound = tags.size, what = "tag")]
            weights[from to to] = input.readDouble()
        }
        return weights
    }

    private fun readFeatureTable(): Pair<List<String>, Map<String, Int>> {
        val count = readCount(what = "feature")
        val features = ArrayList<String>(count)
        val counts = HashMap<String, Int>(count)
        repeat(times = count) {
            val feature = readString()
            features.add(element = feature)
            counts[feature] = input.readInt()
        }
        return features to counts
    }

    private fun readStateWeights(tags: List<Tag>, features: List<String>): Map<Pair<Tag, String>, Double> {
        val count = readCount(what = "state weight")
        val weights = HashMap<Pair<Tag, String>, Double>(count)
        repeat(times = count) {
            val tag = tags[readIndex(bound = tags.size, what = "tag")]
            val feature = features[readIndex(bound = features.size, what = "feature")]
            weights[tag to feature] = input.readDouble()
        }
        return weights
    }

    private fun readCount(what: String): Int {
        val count = input.readInt()
        require(value = count in 0..CrfModelFormat.MAX_ENTRIES) { "implausible $what count $count in SKCR file" }
        return count
    }

    private fun readIndex(bound: Int, what: String): Int {
        val index = input.readInt()
        require(value = index in 0 until bound) { "$what index $index out of range in SKCR file" }
        return index
    }

    private fun readString(): String {
        val length = input.readInt()
        require(value = length in 0..CrfModelFormat.MAX_STRING_BYTES) {
            "implausible string length $length in SKCR file"
        }
        val bytes = ByteArray(size = length)
        input.readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }
}
