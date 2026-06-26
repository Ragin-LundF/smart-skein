import dev.detekt.gradle.Detekt
import dev.detekt.gradle.DetektCreateBaselineTask

plugins {
    id("dev.detekt")
}

val javaVersion = providers.gradleProperty("java_version").get()

tasks.withType<Detekt>().configureEach {
    exclude("**/gen/**")
    reports {
        html.required.set(true) // observe findings in your browser with structure and code snippets
        checkstyle.required.set(true) // checkstyle like format mainly for integrations like Jenkins
        sarif.required.set(true)
        // standardized SARIF format (https://sarifweb.azurewebsites.net/) to support integrations with Github Code Scanning
    }
}

detekt {
    config.setFrom("$rootDir/config/detekt.yml")
    buildUponDefaultConfig = true
}

tasks.withType<Detekt>().configureEach {
    jvmTarget = javaVersion
}
tasks.withType<DetektCreateBaselineTask>().configureEach {
    jvmTarget = javaVersion
}

tasks.matching { it.name == "assemble" }.configureEach {
    finalizedBy(tasks.named("detekt"))
}
