package io.skein.store.postgres.infrastructure

import io.skein.store.postgres.spi.FeatureEncryption
import java.security.SecureRandom
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.SecretKeySpec

/**
 * AES-256-GCM encryption at rest for
 * [io.skein.classify.domain.PrivacyModeEnum.ENCRYPTED_SOURCE]. A fresh random 12-byte IV is
 * generated per encryption and prepended to the ciphertext; GCM's authentication tag detects
 * tampering on decrypt.
 *
 * The 32-byte key is injected by the caller and is never stored in the database.
 */
class AesGcmEncryption(key: ByteArray) : FeatureEncryption {

    init {
        require(value = key.size == AES_256_KEY_BYTES) { "AES-256 requires a 32-byte key, got ${key.size}" }
    }

    private val keySpec = SecretKeySpec(key.copyOf(), "AES")
    private val secureRandom = SecureRandom()

    override fun encrypt(plaintext: ByteArray): ByteArray {
        val iv = ByteArray(size = IV_BYTES)
        secureRandom.nextBytes(iv)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, keySpec, GCMParameterSpec(TAG_BITS, iv))
        val ciphertext = cipher.doFinal(plaintext)
        return iv + ciphertext
    }

    override fun decrypt(ciphertext: ByteArray): ByteArray {
        require(value = ciphertext.size > IV_BYTES) { "ciphertext too short to contain an IV" }
        val iv = ciphertext.copyOfRange(fromIndex = 0, toIndex = IV_BYTES)
        val body = ciphertext.copyOfRange(fromIndex = IV_BYTES, toIndex = ciphertext.size)
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.DECRYPT_MODE, keySpec, GCMParameterSpec(TAG_BITS, iv))
        return cipher.doFinal(body)
    }

    private companion object {
        const val TRANSFORMATION = "AES/GCM/NoPadding"
        const val AES_256_KEY_BYTES = 32
        const val IV_BYTES = 12
        const val TAG_BITS = 128
    }
}
