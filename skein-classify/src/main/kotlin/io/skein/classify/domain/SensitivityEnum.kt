package io.skein.classify.domain

/**
 * Privacy classification of a field. PII fields are excluded from the feature representation
 * so personal data never enters the (already irreversible) hashed feature space in clear text.
 */
enum class SensitivityEnum {
    /** Safe to use as a feature. */
    PUBLIC,

    /** Personal data — never contributes raw content to features. */
    PII,
}
