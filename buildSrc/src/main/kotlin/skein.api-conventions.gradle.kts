import org.jetbrains.kotlin.gradle.dsl.abi.ExperimentalAbiValidation

// Binary-compatibility gate for a published module.
//
// The dump committed under each module's api/ directory is its public ABI. `checkKotlinAbi`
// (wired into `check` below) fails when the current sources no longer match it, and
// `updateKotlinAbi` re-records it. Removing a member, reordering a data class's properties or
// adding one to it all show up there.
//
// This exists because the 1.2.0 -> 2.0.0 break was found by reading a diff. A data class gaining a
// property changes its `copy` and `componentN` signatures, which compiles cleanly, passes every
// test, and only fails at link time in somebody else's build.
//
// Kotlin's own ABI validation rather than the standalone binary-compatibility-validator plugin:
// that plugin reads class files with a bundled ASM that rejects JDK 25's major version 69, the
// same toolchain-lag problem that put this repo on Kover instead of JaCoCo.
plugins {
    id("org.jetbrains.kotlin.jvm")
}

@OptIn(ExperimentalAbiValidation::class)
kotlin {
    abiValidation {
        enabled.set(true)

        filters {
            exclude {
                // Not part of the published surface, and dumping it would be noise on every
                // internal rename.
                byNames.add("io.skein.architecture.**")
            }
        }
    }
}

// `checkKotlinAbi` compares the current ABI against the committed dump; `updateKotlinAbi`
// re-records it. The legacy `*LegacyAbi` tasks exist only to migrate off the standalone plugin and
// are deprecated, so this uses the current format from the start.
tasks.named("check") {
    dependsOn(tasks.named("checkKotlinAbi"))
}
