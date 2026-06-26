package io.skein.classify.domain

/** A record that failed validation during import, together with the reasons it was rejected. */
data class RejectedRecord(val record: Record, val reasons: List<String>)
