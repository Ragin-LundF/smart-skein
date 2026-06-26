plugins {
    kotlin("jvm")
    application
    id("skein.common-conventions")
}

description = "Runnable Skein examples (not published)."

dependencies {
    implementation(project(":skein-text"))
    implementation(project(":skein-classify"))
    implementation(project(":skein-extract"))

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

application {
    mainClass = "io.skein.examples.MainKt"
}
