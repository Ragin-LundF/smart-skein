package io.skein.classify.domain

/**
 * Outcome of validating a [Record] against a [Schema]. Errors mean the record is rejected;
 * warnings are reported but do not block acceptance.
 */
data class ValidationResult(val errors: List<String>, val warnings: List<String>) {

    fun isValid(): Boolean {
        return errors.isEmpty()
    }
}
