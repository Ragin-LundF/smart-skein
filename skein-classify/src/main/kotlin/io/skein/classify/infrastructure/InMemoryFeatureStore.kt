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
 * ponytail: keeps every observation in a list (unbounded). For huge corpora use a paged/persistent
 * adapter such as the PostgreSQL store.
 */
class InMemoryFeatureStore : FeatureStore {

    private val lock = Any()
    private val observations = ArrayList<LabeledFeatures>()

    override fun add(observation: LabeledFeatures) {
        synchronized(lock) {
            observations.add(observation)
        }
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
        }
    }
}
