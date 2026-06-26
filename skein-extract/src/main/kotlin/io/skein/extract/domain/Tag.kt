package io.skein.extract.domain

/** A per-token label produced by a sequence labeler (e.g. `KEY`, `VALUE`, `O`). */
@JvmInline
value class Tag(val value: String) {

    init {
        require(value.isNotBlank()) { "tag must not be blank" }
    }
}
