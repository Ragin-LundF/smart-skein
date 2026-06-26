package io.skein.classify.application

import io.skein.classify.domain.ImportResult
import io.skein.classify.domain.MappedRecord
import io.skein.classify.domain.RejectedRecord
import io.skein.classify.domain.Schema
import io.skein.classify.spi.RecordSource

/**
 * Streams records from a [RecordSource], validates each against the schema and maps the accepted
 * ones into feature-ready [MappedRecord]s. Rejected records are collected with their reasons so
 * import never aborts on a single bad row.
 */
class RecordImportService(
    schema: Schema,
    private val validator: SchemaValidator = SchemaValidator(schema = schema),
    private val mapper: RecordMapper = RecordMapper(schema = schema),
) {

    fun importFrom(source: RecordSource): ImportResult {
        val accepted = mutableListOf<MappedRecord>()
        val rejected = mutableListOf<RejectedRecord>()
        val warnings = mutableListOf<String>()

        source.stream().forEach { record ->
            val result = validator.validate(record = record)
            warnings.addAll(result.warnings)
            if (result.isValid()) {
                accepted.add(mapper.map(record = record))
            } else {
                rejected.add(RejectedRecord(record = record, reasons = result.errors))
            }
        }

        return ImportResult(accepted = accepted, rejected = rejected, warnings = warnings)
    }
}
