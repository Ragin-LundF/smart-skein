package io.skein.extract.infrastructure

/**
 * The `SKCR` on-disk format for a trained [CrfSequenceLabeler].
 *
 * ```
 * offset 0   4 bytes  magic 'S' 'K' 'C' 'R'
 * offset 4   1 byte   format major (breaking)
 * offset 5   1 byte   format minor (additive)
 * offset 6   ...      GZIP( payload )
 * ```
 *
 * The payload is big-endian and strictly sequential: metadata, hyperparameters and step, the tag
 * list, start weights, transitions, an interned feature table with occurrence counts, then state
 * weights. Strings are written as a byte length followed by UTF-8 bytes rather than with
 * `writeUTF`, whose modified-UTF-8 form caps a single string at 65535 bytes.
 *
 * **Evolution contract.** New fields may only be *appended*, and only with a minor bump; any
 * reorder, removal or type change needs a major bump. The reader accepts any minor — including one
 * newer than itself — and never asserts end-of-stream, so a newer writer's extra trailing fields
 * are simply ignored. A differing major is refused outright.
 *
 * Enum values are written as explicit stable codes rather than ordinals, so reordering an enum
 * constant can never silently reinterpret an existing file.
 */
internal object CrfModelFormat {

    val MAGIC = byteArrayOf(0x53, 0x4B, 0x43, 0x52) // 'SKCR'

    const val MAJOR: Int = 1
    const val MINOR: Int = 0

    const val RETENTION_ALL = 0
    const val RETENTION_FREQUENT = 1
    const val RETENTION_STRUCTURAL = 2

    /** Upper bounds validated before allocating, so a corrupt count cannot exhaust memory. */
    const val MAX_ENTRIES = 50_000_000
    const val MAX_STRING_BYTES = 10_000_000

    fun codeOf(retention: FeatureRetentionEnum): Int {
        return when (retention) {
            FeatureRetentionEnum.ALL_FEATURES -> RETENTION_ALL
            FeatureRetentionEnum.FREQUENT_ONLY -> RETENTION_FREQUENT
            FeatureRetentionEnum.STRUCTURAL_ONLY -> RETENTION_STRUCTURAL
        }
    }

    fun retentionOf(code: Int): FeatureRetentionEnum {
        return when (code) {
            RETENTION_ALL -> FeatureRetentionEnum.ALL_FEATURES
            RETENTION_FREQUENT -> FeatureRetentionEnum.FREQUENT_ONLY
            RETENTION_STRUCTURAL -> FeatureRetentionEnum.STRUCTURAL_ONLY
            else -> throw IllegalArgumentException("unknown retention code $code in SKCR model file")
        }
    }
}
