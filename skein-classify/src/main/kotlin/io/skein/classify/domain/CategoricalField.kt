package io.skein.classify.domain

/** Low-cardinality category field; its value is used as a single feature token. */
data class CategoricalField(
    override val name: String,
    override val sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC,
) : FieldSpec
