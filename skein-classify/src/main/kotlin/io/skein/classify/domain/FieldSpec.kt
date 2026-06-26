package io.skein.classify.domain

/**
 * Specification of one field in a [Schema]. Sealed so the schema, validator and mapper can
 * exhaustively reason about every field kind.
 */
sealed interface FieldSpec {

    /** Field name as it appears in a [Record]. */
    val name: String

    /** Privacy classification controlling whether the field may contribute to features. */
    val sensitivity: SensitivityEnum
}
