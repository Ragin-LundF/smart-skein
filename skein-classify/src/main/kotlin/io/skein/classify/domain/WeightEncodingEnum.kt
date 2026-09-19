package io.skein.classify.domain

/**
 * How a fitted weight matrix's values are stored on disk.
 *
 * The weight matrix is around 90% of a persisted multi-label model, so this choice is essentially
 * the choice of file size. Recorded in the file rather than assumed, so the encoding can change
 * without invalidating existing models.
 */
enum class WeightEncodingEnum {

    /** Four bytes per weight, exact. The fallback when a weight is outside [FLOAT16]'s range. */
    FLOAT32,

    /**
     * Two bytes per weight — half the file, for a measured worst-case confidence drift of 0.0004
     * and **zero** changed label decisions across 1.7 million on the reference corpus.
     *
     * Fitted weights sit comfortably within `float16`'s range and far inside the precision that
     * survives pruning, so this is close to free. It is not unconditional: the magnitude ceiling is
     * 65504, and a weight past it would silently become infinity, so the writer verifies the range
     * and falls back to [FLOAT32] rather than corrupting a model.
     */
    FLOAT16,
}
