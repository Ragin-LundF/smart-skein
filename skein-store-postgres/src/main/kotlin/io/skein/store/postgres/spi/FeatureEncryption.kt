package io.skein.store.postgres.spi

/**
 * Encrypts the feature blob before it is written to the database and decrypts it on read.
 * Implementations decide whether data is encrypted at rest; keys are injected into the
 * implementation and never persisted in the database.
 */
interface FeatureEncryption {

    fun encrypt(plaintext: ByteArray): ByteArray

    fun decrypt(ciphertext: ByteArray): ByteArray
}
