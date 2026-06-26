package io.skein.classify.domain

/** Free-text field; contributes its content to character and word n-gram features. */
data class TextField(
    override val name: String,
    override val sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC,
) : FieldSpec
