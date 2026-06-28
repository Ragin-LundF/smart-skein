package io.skein.classify.infrastructure

import io.skein.classify.domain.Label
import io.skein.classify.domain.LabeledFeatures
import io.skein.classify.spi.FeatureStore

/**
 * Default in-memory [FeatureStore]. Suitable for development and bounded corpora.
 *
 * Thread-safe: all access is guarded by an internal lock, and reads return copies, so it is safe to
 * import (write) and train/classify (read) concurrently.
 *
 * Duplicate observations (same label + identical feature vector) are silently discarded so that
 * repeated ingestion of the same source data does not inflate the corpus or the saved model file.
 *
 * ponytail: keeps every unique observation in a list (unbounded). For huge corpora use a
 * paged/persistent adapter such as the PostgreSQL store.
 */
class InMemoryFeatureStore : FeatureStore {

    private val lock = Any()
    private val observations = ArrayList<LabeledFeatures>()
    private val seen = HashSet<Long>()

    override fun add(observation: LabeledFeatures) {
        val fp = fingerprint(observation)
        synchronized(lock) {
            if (seen.add(fp)) observations.add(observation)
        }
    }

    private fun fingerprint(observation: LabeledFeatures): Long {
        var h = observation.label.value.hashCode().toLong()
        h = h * 1_000_003L xor observation.features.indices.contentHashCode().toLong()
        h = h * 1_000_003L xor observation.features.values.contentHashCode().toLong()
        return h
    }

    override fun all(): List<LabeledFeatures> {
        synchronized(lock) {
            return observations.toList()
        }
    }

    override fun labels(): Set<Label> {
        synchronized(lock) {
            return observations.mapTo(destination = LinkedHashSet()) { observation -> observation.label }
        }
    }

    override fun size(): Int {
        synchronized(lock) {
            return observations.size
        }
    }

    override fun clear() {
        synchronized(lock) {
            observations.clear()
            seen.clear()
        }
    }
}
