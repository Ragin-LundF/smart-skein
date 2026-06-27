package io.skein.store.postgres.infrastructure

import javax.crypto.AEADBadTagException
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse

internal class AesGcmEncryptionTest {

    private val key = ByteArray(32) { index -> index.toByte() }
    private val plaintext = "feature-blob-contents".toByteArray()

    @Test
    internal fun `decrypts what it encrypted`() {
        val encryption = AesGcmEncryption(key = key)
        val decrypted = encryption.decrypt(ciphertext = encryption.encrypt(plaintext = plaintext))
        assertContentEquals(expected = plaintext, actual = decrypted)
    }

    @Test
    internal fun `ciphertext differs from plaintext and varies per call`() {
        val encryption = AesGcmEncryption(key = key)
        val first = encryption.encrypt(plaintext = plaintext)
        val second = encryption.encrypt(plaintext = plaintext)
        assertFalse(first.contentEquals(other = plaintext), message = "must not store plaintext")
        assertFalse(first.contentEquals(other = second), message = "random IV must make each ciphertext unique")
    }

    @Test
    internal fun `rejects a tampered ciphertext`() {
        val encryption = AesGcmEncryption(key = key)
        val ciphertext = encryption.encrypt(plaintext = plaintext)
        ciphertext[ciphertext.size - 1] = (ciphertext[ciphertext.size - 1] + 1).toByte()
        assertFailsWith<AEADBadTagException> {
            encryption.decrypt(ciphertext)
        }
    }

    @Test
    internal fun `rejects a key of the wrong size`() {
        assertFailsWith<IllegalArgumentException> {
            AesGcmEncryption(key = ByteArray(size = 16))
        }
    }
}
