import groovy.json.JsonSlurper

// JMH microbenchmarks + performance gate (implementation plan Section 5.4).
// Benchmarks live in src/jmh/kotlin. `benchmarkGate` runs them and fails if any hot path falls
// below its configured throughput floor.
plugins {
    id("me.champeau.jmh")
}

jmh {
    jmhVersion = "1.37"
    fork = 1
    warmupIterations = 1
    iterations = 3
    warmup = "500ms"
    timeOnIteration = "500ms"
    benchmarkMode = listOf("thrpt")
    timeUnit = "s"
    resultFormat = "JSON"
    resultsFile = layout.buildDirectory.file("reports/jmh/results.json")
    failOnError = true
}

tasks.register("benchmarkGate") {
    group = "verification"
    description = "Runs JMH benchmarks and fails if a hot path falls below its throughput floor."
    dependsOn(tasks.named("jmh"))
    doLast {
        val resultsFile = layout.buildDirectory.file("reports/jmh/results.json").get().asFile
        if (!resultsFile.exists()) {
            throw GradleException("JMH results file not found: $resultsFile")
        }
        // Per-benchmark throughput floors (ops/s), keyed by a substring of the benchmark's
        // fully-qualified name. Modules set these via extra["benchmarkThresholds"]. Floors are
        // CALIBRATION KNOBS: deliberately well below observed performance so the gate catches
        // catastrophic regressions (e.g. an accidental O(n^2)) without failing on CI variance.
        @Suppress("UNCHECKED_CAST")
        val thresholds = (project.extra.takeIf { it.has("benchmarkThresholds") }
            ?.get("benchmarkThresholds") as Map<String, Double>?) ?: emptyMap()
        @Suppress("UNCHECKED_CAST")
        val results = JsonSlurper().parse(resultsFile) as List<Map<String, Any>>
        val failures = mutableListOf<String>()
        var checks = 0
        results.forEach { entry ->
            val benchmarkName = entry["benchmark"] as String
            @Suppress("UNCHECKED_CAST")
            val score = ((entry["primaryMetric"] as Map<String, Any>)["score"] as Number).toDouble()
            thresholds.forEach { (key, floor) ->
                if (benchmarkName.contains(key)) {
                    checks++
                    if (score < floor) {
                        failures.add("$benchmarkName: ${String.format("%.0f", score)} ops/s < floor $floor ops/s")
                    }
                }
            }
        }
        if (failures.isNotEmpty()) {
            throw GradleException("Benchmark gate FAILED:\n  " + failures.joinToString("\n  "))
        }
        if (checks == 0) {
            throw GradleException("Benchmark gate ran no threshold checks; configure extra[\"benchmarkThresholds\"]")
        }
        logger.lifecycle("Benchmark gate passed: $checks check(s) across ${results.size} benchmark(s).")
    }
}
