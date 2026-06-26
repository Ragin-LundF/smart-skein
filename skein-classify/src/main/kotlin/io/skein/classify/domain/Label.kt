package io.skein.classify.domain

/**
 * A classification target label. Construction requires a non-blank value; callers validate
 * record content before building a [Label].
 */
@JvmInline
value class Label(val value: String) {

    init {
        require(value = value.isNotBlank()) { "label must not be blank" }
    }
}
