package io.skein.store.postgres.infrastructure

import io.skein.store.postgres.spi.FeatureEncryption

/**
 * Pass-through "encryption" used for [io.skein.classify.domain.PrivacyModeEnum.FEATURES_ONLY]:
 * the stored feature bytes are already an irreversible hash, so no additional encryption is applied.
 */
class NoEncryption : FeatureEncryption {

    override fun encrypt(plaintext: ByteArray): ByteArray {
        return plaintext
    }

    override fun decrypt(ciphertext: ByteArray): ByteArray {
        return ciphertext
    }
}
