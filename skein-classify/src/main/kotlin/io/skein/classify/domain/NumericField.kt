package io.skein.classify.domain

/** Numeric field; its value must be a number or numeric string. */
data class NumericField(
    override val name: String,
    override val sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC,
) : FieldSpec
