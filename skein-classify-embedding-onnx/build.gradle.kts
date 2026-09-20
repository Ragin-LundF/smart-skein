plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.api-conventions")
    id("skein.coverage-conventions")
}

description = "Skein ONNX embedding adapter: implements skein-classify's Vectorizer with an external sentence encoder."

extra["publishName"] = "Skein Classify ONNX Embedding"
extra["publishDescription"] = "Vectorizer adapter backed by an ONNX sentence-embedding model."

dependencies {
    api(project(projectPath = ":skein-classify"))

    // Native binaries live here and nowhere else. A consumer using feature hashing must never
    // inherit an ONNX Runtime shared library it will not load, which is the whole reason this is a
    // separate module rather than an option on skein-classify.
    implementation(libs.onnxruntime)
    implementation(libs.djl.tokenizers)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

// Consumed by the root build's coverage gate. As high as the pure-Kotlin modules: the tiny ONNX
// model and tokenizer committed under src/test/resources mean both native adapters are exercised
// for real, so there is no reason to grant this module a discount. The branches left uncovered are
// failure paths that need a broken native library to reach.
extra["minLineCoverage"] = 92
extra["minBranchCoverage"] = 78
