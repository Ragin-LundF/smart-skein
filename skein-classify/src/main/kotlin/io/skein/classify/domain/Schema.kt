package io.skein.classify.domain

/**
 * Declares the fields of one classification use case. One engine = one schema = one model.
 *
 * Build via the [define] DSL:
 * ```
 * val schema = Schema.define {
 *     text("purpose")
 *     categorical("counterparty")
 *     identifier("iban")
 *     label("category")
 * }
 * ```
 */
class Schema internal constructor(val fields: List<FieldSpec>) {

    /** The single label field; the schema is invalid without exactly one. */
    val labelField: LabelField

    init {
        val labels = fields.filterIsInstance<LabelField>()
        require(value = labels.size == 1) { "schema must declare exactly one label field, found ${labels.size}" }
        labelField = labels.single()
    }

    /** The field named [name], or `null` when not declared. */
    fun field(name: String): FieldSpec? {
        return fields.firstOrNull { spec -> spec.name == name }
    }

    companion object {
        /** Entry point for the schema DSL. */
        fun define(block: SchemaBuilder.() -> Unit): Schema {
            val builder = SchemaBuilder()
            builder.block()
            return builder.build()
        }
    }
}
