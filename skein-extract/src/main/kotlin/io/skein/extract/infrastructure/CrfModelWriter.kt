package io.skein.extract.infrastructure

import io.skein.extract.domain.Tag
import java.io.DataOutputStream

/**
 * Serializes a [CrfModelSnapshot] into the `SKCR` payload described by [CrfModelFormat].
 *
 * Every collection is written in a deterministic order — tags in their semantic order, everything
 * else sorted by index or lexicographically — so saving the same snapshot twice produces identical
 * bytes.
 */
internal class CrfModelWriter(private val output: DataOutputStream) {

    fun write(
        snapshot: CrfModelSnapshot,
        retention: FeatureRetentionEnum,
        minLexicalOccurrences: Int,
        writerVersion: String,
        createdAtEpochMilli: Long,
    ) {
        val retainedState = retain(
            snapshot = snapshot,
            retention = retention,
            minLexicalOccurrences = minLexicalOccurrences,
        )
        val features = retainedState.keys.map { key -> key.second }.distinct().sorted()
        val featureIndices = features.withIndex().associate { (index, feature) -> feature to index }
        val tagIndices = snapshot.tagOrder.withIndex().associate { (index, tag) -> tag to index }

        writeMetadata(
            retention = retention,
            minLexicalOccurrences = minLexicalOccurrences,
            writerVersion = writerVersion,
            createdAtEpochMilli = createdAtEpochMilli,
        )
        writeHyperparameters(snapshot = snapshot)
        writeTags(tags = snapshot.tagOrder)
        writeStartWeights(snapshot = snapshot, tagIndices = tagIndices)
        writeTransitions(snapshot = snapshot, tagIndices = tagIndices)
        writeFeatureTable(features = features, counts = snapshot.featureCounts)
        writeStateWeights(retained = retainedState, tagIndices = tagIndices, featureIndices = featureIndices)
    }

    /**
     * Applies the retention policy. Dropping a weight is equivalent to setting it to zero, because
     * scoring treats a missing state weight as zero — so pruning needs no support in the decoder.
     */
    private fun retain(
        snapshot: CrfModelSnapshot,
        retention: FeatureRetentionEnum,
        minLexicalOccurrences: Int,
    ): Map<Pair<Tag, String>, Double> {
        if (retention == FeatureRetentionEnum.ALL_FEATURES) {
            return snapshot.stateWeights
        }
        return snapshot.stateWeights.filterKeys { key ->
            val feature = key.second
            val lexical = CrfSequenceLabeler.LEXICAL_FEATURE_PREFIXES.any { prefix -> feature.startsWith(prefix) }
            when {
                !lexical -> true
                retention == FeatureRetentionEnum.STRUCTURAL_ONLY -> false
                else -> (snapshot.featureCounts[feature] ?: 0) >= minLexicalOccurrences
            }
        }
    }

    private fun writeMetadata(
        retention: FeatureRetentionEnum,
        minLexicalOccurrences: Int,
        writerVersion: String,
        createdAtEpochMilli: Long,
    ) {
        output.writeLong(createdAtEpochMilli)
        writeString(value = writerVersion)
        output.writeInt(CrfModelFormat.codeOf(retention = retention))
        output.writeInt(minLexicalOccurrences)
    }

    private fun writeHyperparameters(snapshot: CrfModelSnapshot) {
        output.writeDouble(snapshot.initialLearningRate)
        output.writeDouble(snapshot.decayRate)
        output.writeDouble(snapshot.l2Regularization)
        output.writeLong(snapshot.step)
    }

    private fun writeTags(tags: List<Tag>) {
        output.writeInt(tags.size)
        tags.forEach { tag -> writeString(value = tag.value) }
    }

    private fun writeStartWeights(snapshot: CrfModelSnapshot, tagIndices: Map<Tag, Int>) {
        val entries = snapshot.startWeights.entries.sortedBy { entry -> tagIndices.getValue(entry.key) }
        output.writeInt(entries.size)
        entries.forEach { entry ->
            output.writeInt(tagIndices.getValue(entry.key))
            output.writeDouble(entry.value)
        }
    }

    private fun writeTransitions(snapshot: CrfModelSnapshot, tagIndices: Map<Tag, Int>) {
        val entries = snapshot.transitionWeights.entries.sortedWith(
            compareBy(
                { entry -> tagIndices.getValue(entry.key.first) },
                { entry -> tagIndices.getValue(entry.key.second) },
            ),
        )
        output.writeInt(entries.size)
        entries.forEach { entry ->
            output.writeInt(tagIndices.getValue(entry.key.first))
            output.writeInt(tagIndices.getValue(entry.key.second))
            output.writeDouble(entry.value)
        }
    }

    private fun writeFeatureTable(features: List<String>, counts: Map<String, Int>) {
        output.writeInt(features.size)
        features.forEach { feature ->
            writeString(value = feature)
            output.writeInt(counts[feature] ?: 0)
        }
    }

    private fun writeStateWeights(
        retained: Map<Pair<Tag, String>, Double>,
        tagIndices: Map<Tag, Int>,
        featureIndices: Map<String, Int>,
    ) {
        val entries = retained.entries.sortedWith(
            compareBy(
                { entry -> tagIndices.getValue(entry.key.first) },
                { entry -> featureIndices.getValue(entry.key.second) },
            ),
        )
        output.writeInt(entries.size)
        entries.forEach { entry ->
            output.writeInt(tagIndices.getValue(entry.key.first))
            output.writeInt(featureIndices.getValue(entry.key.second))
            output.writeDouble(entry.value)
        }
    }

    private fun writeString(value: String) {
        val bytes = value.toByteArray(charset = Charsets.UTF_8)
        output.writeInt(bytes.size)
        output.write(bytes)
    }
}
