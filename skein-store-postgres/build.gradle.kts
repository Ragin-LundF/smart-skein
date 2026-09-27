plugins {
    kotlin("jvm")
    `java-library`
    id("skein.published-library-conventions")
    id("skein.coverage-conventions")
}

description = "Skein PostgreSQL persistence adapter: implements skein-classify's FeatureStore on PostgreSQL."

extra["publishName"] = "Skein Store PostgreSQL"
extra["publishDescription"] = "PostgreSQL FeatureStore adapter with optional AES-256-GCM encryption at rest."

dependencies {
    api(project(projectPath = ":skein-classify"))
    implementation(libs.tomcat.jdbc)
    implementation(libs.liquibase.core)
    runtimeOnly(libs.postgresql)

    testImplementation(libs.kotlin.test.junit5)
    testImplementation(libs.testcontainers.postgresql)
    testImplementation(libs.testcontainers.junit)
    testRuntimeOnly(libs.junit.jupiter)
    testRuntimeOnly(libs.postgresql)
}

// Consumed by the root build's coverage gate.
extra["minLineCoverage"] = 90
extra["minBranchCoverage"] = 75
