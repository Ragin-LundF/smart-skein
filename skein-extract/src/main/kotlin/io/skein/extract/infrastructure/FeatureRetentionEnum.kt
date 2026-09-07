package io.skein.extract.infrastructure

/**
 * How much of a CRF's learned vocabulary is written when a model is saved.
 *
 * This is a privacy control, not a size control. A CRF learns features keyed by the actual token
 * text (`word=`, and three-character `prefix=`/`suffix=` affixes), so a saved model contains
 * fragments of the training corpus in clear — unlike `skein-classify`, whose features are
 * irreversible hashes.
 */
enum class FeatureRetentionEnum {

    /**
     * Write every weight. The only mode that restores a bit-identical model, and the only one that
     * writes every lexical feature — including anything seen just once, which is exactly the shape
     * of a name, an IBAN or an account reference.
     */
    ALL_FEATURES,

    /**
     * Default. Drop lexical features seen fewer than the configured number of times, keeping all
     * structural features and every transition and start weight.
     *
     * This **reduces** exposure rather than eliminating it: a name occurring twice in the corpus
     * survives. A feature updated from a single occurrence is also statistically worthless, so the
     * accuracy cost is small.
     */
    FREQUENT_ONLY,

    /**
     * Write no lexical features at all — only token-type structure. The single mode that carries a
     * zero-clear-text guarantee. It costs the model its ability to generalize from word shape, such
     * as recognizing an unseen `-ing` word from its suffix.
     */
    STRUCTURAL_ONLY,
}
