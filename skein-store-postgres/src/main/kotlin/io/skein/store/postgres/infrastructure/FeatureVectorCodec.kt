package io.skein.store.postgres.infrastructure

import io.skein.classify.domain.FeatureVector
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.DataInputStream
import java.io.DataOutputStream

/**
 * Binary serialization of a [FeatureVector] for `BYTEA` storage.
 *
 * Layout: `int count`, then `count` big-endian ints (indices), then `count` big-endian floats
 * (values). Compact and allocation-light; round-trips exactly.
 */
class FeatureVectorCodec {

    fun encode(vector: FeatureVector): ByteArray {
        val buffer = ByteArrayOutputStream()
        DataOutputStream(buffer).use { output ->
            output.writeInt(vector.indices.size)
            for (index in vector.indices) {
                output.writeInt(index)
            }
            for (value in vector.values) {
                output.writeFloat(value)
            }
        }
        return buffer.toByteArray()
    }

    fun decode(bytes: ByteArray): FeatureVector {
        DataInputStream(ByteArrayInputStream(bytes)).use { input ->
            val count = input.readInt()
            val indices = IntArray(count) { input.readInt() }
            val values = FloatArray(count) { input.readFloat() }
            return FeatureVector(indices = indices, values = values)
        }
    }
}
