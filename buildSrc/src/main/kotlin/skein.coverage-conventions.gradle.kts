import kotlinx.kover.gradle.plugin.dsl.CoverageUnit

// Per-module coverage gate (implementation plan Section 7.1): line + branch, enforced in the build.
// Kover is used instead of JaCoCo because its IntelliJ coverage agent tracks current Kotlin/JVM
// versions (JaCoCo's class-file support lags behind JDK 25).
plugins {
    id("org.jetbrains.kotlinx.kover")
}

// Per-module thresholds (% covered). Calibration knobs: a module raises them via `extra` in its
// build script (read lazily, so overrides set after this plugin applies are honored). Defaults are
// conservative floors that catch coverage regressions.
val minLine = providers.provider { (extra.takeIf { it.has("minLineCoverage") }?.get("minLineCoverage") as Int?) ?: 70 }
val minBranch = providers.provider { (extra.takeIf { it.has("minBranchCoverage") }?.get("minBranchCoverage") as Int?) ?: 50 }

kover {
    reports {
        filters {
            excludes {
                classes("*Benchmark")
            }
        }
        verify {
            rule {
                bound {
                    minValue = minLine
                    coverageUnits = CoverageUnit.LINE
                }
                bound {
                    minValue = minBranch
                    coverageUnits = CoverageUnit.BRANCH
                }
            }
        }
    }
}

tasks.named("check") {
    dependsOn(tasks.named("koverVerify"))
}
