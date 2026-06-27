@file:Suppress("DEPRECATION") // JReleaser 1.24 deprecates a few signing accessors that are still functional.

import org.gradle.api.component.SoftwareComponent
import org.gradle.api.publish.maven.MavenPublication

plugins {
    `maven-publish`
    id("org.jreleaser")
}

afterEvaluate {
    tasks.matching { it.name.startsWith("publish") }.configureEach {
        dependsOn(tasks.named("dokkaGenerateHtmlJar"))
    }
    tasks.matching { it.name.contains("MetadataFileFor") || it.name.contains("PomFileFor") }.configureEach {
        dependsOn(tasks.named("dokkaGenerateHtmlJar"))
    }
}

fun configurePublication(publicationComponent: SoftwareComponent) {
    publishing {
        publications {
            create<MavenPublication>("maven") {
                from(publicationComponent)

                artifactId = if (extra.has("publishArtifactId")) extra["publishArtifactId"] as String else project.name
                version = project.version.toString()
                // Add Dokka-generated Javadoc jar (lazy reference)
                artifact(provider { tasks.named("dokkaGenerateHtmlJar").get() })
                pom {
                    name.set(if (extra.has("publishName")) extra["publishName"] as String else project.name)
                    description.set(if (extra.has("publishDescription")) extra["publishDescription"] as String else "Skein Library")
                    url.set("https://github.com/Ragin-LundF/smart-skein")
                    scm {
                        connection.set("scm:git:git@github.com:Ragin-LundF/smart-skein.git")
                        developerConnection.set("scm:git:ssh://github.com/Ragin-LundF/smart-skein.git")
                        url.set("https://github.com/Ragin-LundF/smart-skein")
                    }
                    licenses {
                        license {
                            name.set("MIT")
                            url.set("https://github.com/Ragin-LundF/smart-skein/blob/main/LICENSE")
                        }
                    }
                    developers {
                        developer {
                            id.set("Ragin-LundF")
                            name.set("Ragin-LundF")
                            organizationUrl.set("https://github.com/Ragin-LundF/smart-skein")
                        }
                    }
                }
            }
        }

        repositories {
            maven {
                setUrl(layout.buildDirectory.dir("staging-deploy"))
            }
            maven {
                name = "GitHubPackages"
                url = uri("https://maven.pkg.github.com/Ragin-LundF/smart-skein")
                credentials {
                    username = System.getenv("GITHUB_ACTOR")
                    password = System.getenv("GITHUB_TOKEN")
                }
            }
        }
    }
}

// BOM/Platform module
pluginManager.withPlugin("java-platform") {
    configurePublication(components.getByName("javaPlatform"))
}

// java-library modules
pluginManager.withPlugin("java") {
    configurePublication(components.getByName("java"))
}

jreleaser {
    signing {
        setActive("ALWAYS")
        verify.set(false)
        armored.set(true)
        setMode("FILE")
        publicKey.set("$rootDir/ragin_public.pgp")
        secretKey.set("$rootDir/ragin_secret.pgp")
    }
    deploy {
        maven {
            mavenCentral {
                create("sonatype") {
                    setActive("ALWAYS")
                    url.set("https://central.sonatype.com/api/v1/publisher")
                    stagingRepository("build/staging-deploy")
                }
            }
        }
    }
}
