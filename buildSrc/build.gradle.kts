plugins {
    `kotlin-dsl`
}

// Plugin artifacts the convention scripts in src/main/kotlin compile and apply against.
// Versions come from the root version catalog so they stay in sync with the modules.
fun pluginDep(plugin: Provider<PluginDependency>) =
    plugin.map { "${it.pluginId}:${it.pluginId}.gradle.plugin:${it.version}" }

dependencies {
    implementation(pluginDep(libs.plugins.kotlin.jvm))
    implementation(pluginDep(libs.plugins.detekt))
    implementation(pluginDep(libs.plugins.kotlin.dokka))
    implementation(pluginDep(libs.plugins.jmh))
    implementation(pluginDep(libs.plugins.kover))
    implementation(pluginDep(libs.plugins.jreleaser))
}
