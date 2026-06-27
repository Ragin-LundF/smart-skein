import org.jetbrains.dokka.gradle.engine.plugins.DokkaHtmlPluginParameters

plugins {
    id("org.jetbrains.dokka")
}

dokka {
    pluginsConfiguration.withType<DokkaHtmlPluginParameters>().configureEach {
        footerMessage.set("(c) Ragin https://github.com/Ragin-LundF/smart-skein")
        separateInheritedMembers.set(false)
        mergeImplicitExpectActualDeclarations.set(false)
    }
}

if (tasks.findByName("dokkaGenerateHtmlJar") == null) {
    tasks.register<Jar>("dokkaGenerateHtmlJar") {
        group = "documentation"
        description = "Assembles a Javadoc JAR from Dokka HTML output"
        dependsOn(tasks.named("dokkaGenerateHtml"))
        archiveClassifier.set("javadoc")
        from(layout.buildDirectory.dir("dokka/html"))
    }
}
