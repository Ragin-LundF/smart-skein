plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.coverage-conventions")
}

description = "Skein extraction: pull structured values out of text via typed-token patterns and slot filling."

extra["publishName"] = "Skein Extract"
extra["publishDescription"] = "Text → structured fields via typed-token patterns and slot filling."

dependencies {
    api(project(projectPath = ":skein-text"))

    testImplementation(libs.kotlin.test.junit5)
    testRuntimeOnly(libs.junit.jupiter)
}

// Consumed by the root build's coverage gate.
extra["minLineCoverage"] = 90
extra["minBranchCoverage"] = 80
