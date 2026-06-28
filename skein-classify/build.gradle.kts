plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.benchmark-conventions")
    id("skein.coverage-conventions")
    alias(libs.plugins.kotlin.serialization)
}

description = "Skein classification: record → label, typo-tolerant, self-training, privacy-preserving."

extra["publishName"] = "Skein Classify"
extra["publishDescription"] = "Record → label classification with privacy-preserving feature hashing."

dependencies {
    api(project(":skein-text"))
    implementation(libs.kotlinx.serialization.protobuf)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

// Consumed by the root build's coverage + benchmark gates.
extra["benchmarkThresholds"] = mapOf(
    "HashingBenchmark.sipHash" to 1000000.0,
    "VectorizationBenchmark.vectorize" to 2000.0,
    "ScoringBenchmark.classify" to 2000.0,
)
extra["minLineCoverage"] = 92
extra["minBranchCoverage"] = 72
