package io.skein.classify.domain

/** How active learning measures how uncertain a prediction is (higher = more worth labeling). */
enum class UncertaintyStrategyEnum {
    /** Top-two probability gap; the smaller the gap, the more uncertain. The default. */
    MARGIN,

    /** `1 - confidence` of the top label. */
    LEAST_CONFIDENCE,

    /** Shannon entropy over all label probabilities; peaks when the distribution is flat. */
    ENTROPY,
}
