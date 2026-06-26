package io.skein.extract.domain

/** All fields extracted from one piece of text. Persists nothing; it simply carries the values. */
data class ExtractionResult(val fields: List<ExtractedField>) {

    /** All values extracted for [name], in source order (more than one for repeating groups). */
    fun valuesOf(name: String): List<String> {
        return fields.filter { field -> field.name == name }.map { field -> field.value }
    }

    /** The first field named [name], or `null` when nothing matched. */
    fun first(name: String): ExtractedField? {
        return fields.firstOrNull { field -> field.name == name }
    }
}
