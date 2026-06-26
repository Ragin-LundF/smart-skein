package io.skein.classify.domain

/**
 * Identifier field (customer number, IBAN, ...). Defaults to [SensitivityEnum.PII] and is
 * therefore excluded from the raw feature content by default.
 */
data class IdentifierField(
    override val name: String,
    override val sensitivity: SensitivityEnum = SensitivityEnum.PII,
) : FieldSpec
