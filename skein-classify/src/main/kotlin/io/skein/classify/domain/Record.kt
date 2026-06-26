package io.skein.classify.domain

/**
 * Input contract for classification: a thin, immutable wrapper over a field map.
 *
 * Core stays map-based on purpose; annotation-based mapping of user data classes is a later
 * convenience adapter, not a core concern.
 */
@JvmInline
value class Record(val values: Map<String, Any?>) {

    /** Raw value of [field], or `null` when absent. */
    operator fun get(field: String): Any? {
        return values[field]
    }

    /** All field names present in this record. */
    fun fieldNames(): Set<String> {
        return values.keys
    }
}
