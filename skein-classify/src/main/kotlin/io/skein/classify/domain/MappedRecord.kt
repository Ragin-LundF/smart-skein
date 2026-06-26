package io.skein.classify.domain

/**
 * A record reduced to what classification needs: the concatenated feature text (PII fields
 * excluded) and the extracted [label], which is `null` for records to be classified rather
 * than learned from.
 */
data class MappedRecord(val featureText: String, val label: Label?)
