plugins {
    kotlin("jvm")
    application
    id("skein.common-conventions")
    alias(libs.plugins.kotlin.serialization)
}

description = "Runnable Skein examples (not published)."

dependencies {
    implementation(project(":skein-text"))
    implementation(project(":skein-classify"))
    implementation(project(":skein-extract"))
    implementation(project(":skein-classify-embedding-onnx"))
    // The recipe example's ruleset is JSON, to make the point that the rule format is the
    // caller's business and never the library's. Not published, so nothing inherits this.
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

application {
    mainClass = "io.skein.examples.MainKt"
}
