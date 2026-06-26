package io.skein.extract.spi

import io.skein.extract.domain.Tag
import io.skein.text.domain.Token

/**
 * Port for a learnable token tagger: assigns one [Tag] to each [Token] in a sequence, learning the
 * regularities (token features and tag transitions) from labeled examples.
 */
interface SequenceLabeler {

    /** Incrementally learns from one labeled sequence; [tokens] and [tags] must have equal length. */
    fun learn(tokens: List<Token>, tags: List<Tag>)

    /** Predicts the most likely tag sequence for [tokens]. Requires prior training. */
    fun label(tokens: List<Token>): List<Tag>
}
