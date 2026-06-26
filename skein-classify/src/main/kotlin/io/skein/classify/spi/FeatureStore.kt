package io.skein.classify.spi

import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures

/**
 * Persistence port for the privacy-preserving training corpus: hashed, irreversible feature
 * vectors paired with their labels. Adapters (in-memory here, PostgreSQL in `skein-store-postgres`)
 * implement it. Classifiers build or update their model from what the store holds.
 */
interface FeatureStore {

    /** Persists one labeled observation. */
    fun add(observation: LabeledFeatures)

    /** Persists many labeled observations. */
    fun addAll(observations: Collection<LabeledFeatures>) {
        observations.forEach { observation -> add(observation) }
    }

    /** All stored observations. */
    fun all(): List<LabeledFeatures>

    /** Distinct labels seen so far. */
    fun labels(): Set<Label>

    /** Number of stored observations. */
    fun size(): Int

    /** Removes every stored observation (used by `forget`). */
    fun clear()
}
