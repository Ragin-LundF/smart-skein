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
    implementation(project(":skein-classify-embedding-http"))
    // The recipe example's ruleset is JSON, to make the point that the rule format is the
    // caller's business and never the library's. Not published, so nothing inherits this.
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

application {
    mainClass = "io.skein.examples.MainKt"
}

// `./gradlew :examples:run -Dskein.onnx.model=...` sets the property on the Gradle daemon's JVM,
// not on the forked one this task starts, so without forwarding it an example never sees it and
// prints its "not configured" help instead -- including when the user followed the instructions
// that example itself printed. Everything under `skein.` is forwarded, so a new example can read
// a new property without touching this file.
tasks.named<JavaExec>("run") {
    systemProperties(providers.systemPropertiesPrefixedBy("skein.").get())
}
