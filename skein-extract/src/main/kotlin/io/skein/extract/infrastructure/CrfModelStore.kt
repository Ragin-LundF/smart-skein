package io.skein.extract.infrastructure

import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.EOFException
import java.nio.file.Path
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import java.util.zip.ZipException
import kotlin.io.path.inputStream
import kotlin.io.path.outputStream

/** Header bytes preceding the compressed payload: 4 magic plus two version bytes. */
private const val HEADER_SIZE = 6

/**
 * Saves and loads a trained [CrfSequenceLabeler] as a single `SKCR` file.
 *
 * ### Privacy — read before saving
 *
 * A CRF learns features keyed by the actual token text, so a saved model contains fragments of the
 * training corpus **in clear**. This is unlike `skein-classify`, whose features are irreversible
 * hashes. Anyone holding the file can decompress it and read the vocabulary.
 *
 * The default [FeatureRetentionEnum.FREQUENT_ONLY] drops lexical features seen fewer than
 * [DEFAULT_MIN_LEXICAL_OCCURRENCES] times, which removes the once-seen strings most likely to be
 * personal data. It **reduces exposure rather than eliminating it** — a name appearing twice
 * survives. Only [FeatureRetentionEnum.STRUCTURAL_ONLY] carries a zero-clear-text guarantee, and
 * only [FeatureRetentionEnum.ALL_FEATURES] restores a bit-identical model.
 *
 * ponytail: this file operates on a snapshot rather than through a port, because there is exactly
 * one sequence labeler with weights to persist. Upgrade path: promote snapshot/restore to a
 * capability interface in `spi` when a second implementation appears — [SequenceLabeler] itself
 * stays free of persistence, since a rule-based labeler has no weights to write.
 */
object CrfModelStore {

    /** Occurrences a lexical feature needs before [FeatureRetentionEnum.FREQUENT_ONLY] keeps it. */
    const val DEFAULT_MIN_LEXICAL_OCCURRENCES = 2

    fun save(
        path: Path,
        snapshot: CrfModelSnapshot,
        retention: FeatureRetentionEnum = FeatureRetentionEnum.FREQUENT_ONLY,
        minLexicalOccurrences: Int = DEFAULT_MIN_LEXICAL_OCCURRENCES,
    ) {
        require(value = snapshot.tagOrder.isNotEmpty()) { "cannot save an untrained labeler" }
        require(value = minLexicalOccurrences >= 1) { "minLexicalOccurrences must be at least 1" }
        path.outputStream().use { file ->
            file.write(CrfModelFormat.MAGIC)
            file.write(CrfModelFormat.MAJOR)
            file.write(CrfModelFormat.MINOR)
            GZIPOutputStream(file).use { gzip ->
                val output = DataOutputStream(gzip)
                CrfModelWriter(output = output).write(
                    snapshot = snapshot,
                    retention = retention,
                    minLexicalOccurrences = minLexicalOccurrences,
                    writerVersion = writerVersion(),
                    createdAtEpochMilli = System.currentTimeMillis(),
                )
                output.flush()
            }
        }
    }

    fun load(path: Path): CrfModelSnapshot {
        return read(path = path) { reader, major, minor ->
            reader.readMetadata(formatMajor = major, formatMinor = minor)
            reader.readSnapshot()
        }
    }

    /** Reads only the header and metadata, without inflating the weights. */
    fun metadata(path: Path): CrfModelMetadata {
        return read(path = path) { reader, major, minor ->
            reader.readMetadata(formatMajor = major, formatMinor = minor)
        }
    }

    private fun <T> read(path: Path, body: (CrfModelReader, Int, Int) -> T): T {
        return path.inputStream().use { file ->
            val header = file.readNBytes(HEADER_SIZE)
            require(
                value = header.size == HEADER_SIZE &&
                    CrfModelFormat.MAGIC.indices.all { index -> header[index] == CrfModelFormat.MAGIC[index] },
            ) { "not a SKCR model file" }
            val major = header[4].toInt()
            val minor = header[5].toInt()
            require(value = major == CrfModelFormat.MAJOR) {
                "SKCR format major $major cannot be read by this build, which reads major ${CrfModelFormat.MAJOR}"
            }
            // Any minor is accepted, including a newer one: the reader is sequential and never
            // asserts end-of-stream, so fields a newer writer appended are simply skipped.
            try {
                GZIPInputStream(file).use { gzip -> body(CrfModelReader(input = DataInputStream(gzip)), major, minor) }
            } catch (cause: EOFException) {
                throw IllegalArgumentException("truncated SKCR model file", cause)
            } catch (cause: ZipException) {
                throw IllegalArgumentException("corrupt SKCR model file", cause)
            }
        }
    }

    // ponytail: reads "unknown" when running from Gradle class directories rather than a published
    // jar. Upgrade path: a generated build constant.
    private fun writerVersion(): String {
        return CrfModelStore::class.java.`package`?.implementationVersion ?: "unknown"
    }
}
