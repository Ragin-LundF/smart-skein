plugins {
    kotlin("jvm")
    application
    id("skein.published-library-conventions")
    alias(libs.plugins.kotlin.serialization)
}

description = "Skein CLI: command-line tools for training and inspecting classifiers."

extra["publishName"] = "Skein CLI"
extra["publishDescription"] = "Command-line tools for active-learning data labeling and classification."

dependencies {
    implementation(project(":skein-classify"))
    implementation(libs.kotlinx.serialization.protobuf)

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

application {
    mainClass = "io.skein.cli.MainKt"
}
