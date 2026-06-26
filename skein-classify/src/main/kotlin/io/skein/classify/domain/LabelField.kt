package io.skein.classify.domain

/** The classification target field. A schema must declare exactly one. */
data class LabelField(
    override val name: String,
    override val sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC,
) : FieldSpec
