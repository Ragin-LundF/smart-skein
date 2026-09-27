plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.benchmark-conventions")
    id("skein.coverage-conventions")
}

description = "Skein shared text foundation: normalization, typed tokenizer, pattern signatures."

extra["publishName"] = "Skein Text"
extra["publishDescription"] = "Shared text foundation: normalization, typed tokenizer, pattern signatures."

dependencies {
    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

// Consumed by the root build's coverage + benchmark gates.
extra["benchmarkThresholds"] = mapOf(
    "TokenizationBenchmark.tokenize" to 50000.0,
)
extra["minLineCoverage"] = 85
extra["minBranchCoverage"] = 85
