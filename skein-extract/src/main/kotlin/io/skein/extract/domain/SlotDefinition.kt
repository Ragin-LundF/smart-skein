package io.skein.extract.domain

/** A rule describing how to extract one named value (or repeating group) from a token stream. */
sealed interface SlotDefinition {

    /** Slot name; for repeating groups this is the group identity, the values use the component names. */
    val name: String
}
