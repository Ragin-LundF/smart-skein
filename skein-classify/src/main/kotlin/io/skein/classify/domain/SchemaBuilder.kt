package io.skein.classify.domain

/**
 * Mutable builder backing [Schema.define]. Each method appends one field spec; [build] freezes
 * the collected fields into an immutable [Schema].
 */
class SchemaBuilder {

    private val fields = ArrayList<FieldSpec>()

    fun text(name: String, sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC) {
        fields.add(TextField(name = name, sensitivity = sensitivity))
    }

    fun categorical(name: String, sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC) {
        fields.add(CategoricalField(name = name, sensitivity = sensitivity))
    }

    fun numeric(name: String, sensitivity: SensitivityEnum = SensitivityEnum.PUBLIC) {
        fields.add(NumericField(name = name, sensitivity = sensitivity))
    }

    fun identifier(name: String, sensitivity: SensitivityEnum = SensitivityEnum.PII) {
        fields.add(IdentifierField(name = name, sensitivity = sensitivity))
    }

    fun label(name: String) {
        fields.add(LabelField(name = name))
    }

    fun build(): Schema {
        return Schema(fields.toList())
    }
}
