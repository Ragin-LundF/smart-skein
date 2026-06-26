package io.skein.classify.spi

import io.skein.classify.domain.Record

/**
 * Source of raw records to import. Returns a [Sequence] so large inputs stream with constant
 * memory rather than being materialized up front.
 */
interface RecordSource {

    fun stream(): Sequence<Record>
}
