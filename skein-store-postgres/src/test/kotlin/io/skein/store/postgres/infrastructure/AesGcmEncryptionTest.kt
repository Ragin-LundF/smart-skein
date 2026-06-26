package io.skein.store.postgres.infrastructure

import javax.crypto.AEADBadTagException
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

class AesGcmEncryptionTest {

    private val key = ByteArray(32) { index -> index.toByte() }
    private val plaintext = "feature-blob-contents".toByteArray()

    @Test
    fun `decrypts what it encrypted`() {
        val encryption = AesGcmEncryption(key)
        val decrypted = encryption.decrypt(encryption.encrypt(plaintext))
        assertContentEquals(expected = plaintext, actual = decrypted)
    }

    @Test
    fun `ciphertext differs from plaintext and varies per call`() {
        val encryption = AesGcmEncryption(key)
        val first = encryption.encrypt(plaintext)
        val second = encryption.encrypt(plaintext)
        assertFalse(first.contentEquals(plaintext), message = "must not store plaintext")
        assertFalse(first.contentEquals(second), message = "random IV must make each ciphertext unique")
    }

    @Test
    fun `rejects a tampered ciphertext`() {
        val encryption = AesGcmEncryption(key)
        val ciphertext = encryption.encrypt(plaintext)
        ciphertext[ciphertext.size - 1] = (ciphertext[ciphertext.size - 1] + 1).toByte()
        assertFailsWith<AEADBadTagException> {
            encryption.decrypt(ciphertext)
        }
    }

    @Test
    fun `rejects a key of the wrong size`() {
        assertFailsWith<IllegalArgumentException> {
            AesGcmEncryption(ByteArray(16))
        }
    }
}
