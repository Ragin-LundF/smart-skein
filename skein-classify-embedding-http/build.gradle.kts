plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.api-conventions")
    id("skein.coverage-conventions")
    alias(libs.plugins.kotlin.serialization)
}

description =
    "Skein HTTP embedding adapter: implements skein-classify's Vectorizer against an OpenAI-compatible /embeddings endpoint."

extra["publishName"] = "Skein Classify HTTP Embedding"
extra["publishDescription"] = "Vectorizer adapter backed by an OpenAI-compatible embeddings service."

dependencies {
    api(project(projectPath = ":skein-classify"))

    // skein-classify keeps this at test scope, because its model format is ProtoBuf and a consumer
    // doing feature hashing should not inherit a JSON parser it never calls. That reasoning is
    // about that module, not this one: the OpenAI embeddings protocol *is* JSON, so anyone adding
    // this adapter has asked for a JSON client by definition. Do not "fix" this to match the
    // sibling comment. It is `implementation` rather than `api`, so it stays off the consumer's
    // compile classpath either way.
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

// Consumed by the root build's coverage gate. One notch above the ONNX adapter's branch floor:
// that module discounts itself for "failure paths that need a broken native library to reach",
// and this one has no such excuse. Every branch here -- non-200, wrong entry count, wrong width,
// malformed JSON, an unprobed dimension, canary drift -- is reachable from a local test server or
// a fake transport.
extra["minLineCoverage"] = 92
extra["minBranchCoverage"] = 80
