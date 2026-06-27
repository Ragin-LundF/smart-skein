package io.skein.architecture

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Clean-architecture layering gate (implementation plan §4.3 / §7.2): dependencies must flow
 * inward. This scans every module's production Kotlin sources and fails on a forbidden import.
 *
 * - `domain` depends on nothing outside the domain (the plan's "no domain → infrastructure" rule,
 *   plus no domain → application/spi).
 * - `spi` (ports) depend only on the domain.
 *
 * It is intentionally dependency-free (no architecture library) so it cannot break on toolchain
 * upgrades; imports in Kotlin are line-oriented, which is all this rule needs.
 */
internal class LayeringArchitectureTest {

    @Test
    internal fun `domain packages depend on nothing outside the domain`() {
        val violations = violations(
            layerSuffix = ".domain",
            forbidden = listOf(".application.", ".spi.", ".infrastructure."),
        )
        assertTrue(
            actual = violations.isEmpty(),
            message = "domain must not depend on other layers:\n${violations.joinToString(separator = "\n")}",
        )
    }

    @Test
    internal fun `spi ports depend only on the domain`() {
        val violations = violations(layerSuffix = ".spi", forbidden = listOf(".application.", ".infrastructure."))
        assertTrue(
            actual = violations.isEmpty(),
            message = "spi ports must depend only on the domain:\n${violations.joinToString(separator = "\n")}",
        )
    }

    @Test
    internal fun `the scan actually inspects domain sources`() {
        // Guard against a vacuous pass if source discovery ever breaks.
        val domainFiles = productionKotlinFiles()
            .filter { file -> packageOf(lines = file.readLines())?.endsWith(suffix = ".domain") == true }
        assertTrue(
            actual = domainFiles.size >= EXPECTED_MINIMUM_DOMAIN_FILES,
            message = "expected to find domain sources to check",
        )
    }

    private fun violations(layerSuffix: String, forbidden: List<String>): List<String> {
        return productionKotlinFiles().flatMap { file ->
            val lines = file.readLines()
            val packageName = packageOf(lines = lines)
            if (packageName == null || !packageName.endsWith(suffix = layerSuffix)) {
                emptyList()
            } else {
                forbiddenImports(lines = lines, forbidden = forbidden)
                    .map { import -> "${file.name} ($packageName) imports $import" }
            }
        }
    }

    private fun packageOf(lines: List<String>): String? {
        return lines.firstOrNull { line -> line.trimStart().startsWith(prefix = "package ") }
            ?.substringAfter(delimiter = "package ")
            ?.trim()
    }

    private fun forbiddenImports(lines: List<String>, forbidden: List<String>): List<String> {
        return lines
            .filter { line -> line.trimStart().startsWith(prefix = "import ") }
            .map { line -> line.substringAfter(delimiter = "import ").trim() }
            .filter { import ->
                import.startsWith(prefix = ROOT_PACKAGE) && forbidden.any { layer -> import.contains(other = layer) }
            }
    }

    private fun productionKotlinFiles(): List<File> {
        return repositoryRoot().walkTopDown()
            .filter { file -> file.isFile && file.extension == "kt" }
            .filter { file -> file.path.replace(File.separatorChar, '/').contains("/src/main/kotlin/") }
            .toList()
    }

    private fun repositoryRoot(): File {
        var current: File? = File(System.getProperty("user.dir"))
        while (current != null) {
            if (File(current, "settings.gradle").exists() || File(current, "settings.gradle.kts").exists()) {
                return current
            }
            current = current.parentFile
        }
        error(message = "could not locate repository root (settings.gradle)")
    }

    private companion object {
        const val ROOT_PACKAGE = "io.skein"
        const val EXPECTED_MINIMUM_DOMAIN_FILES = 5
    }
}
