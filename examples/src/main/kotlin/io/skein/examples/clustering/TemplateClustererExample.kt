package io.skein.examples.clustering

import io.skein.extract.application.TemplateClusterer

fun runTemplateClustererExample() {
    val corpus = listOf(
        "AIG-Life 67,89 insurance premium",
        "Geico-Auto 120,00 insurance premium",
        "Allstate-Home 90,00 insurance premium",
        "rent apartment CustomerNumber AB123 monthly",
        "rent apartment CustomerNumber XY999 monthly",
        "rent apartment CustomerNumber CD456 monthly",
        "salary 2024-03-01",
        "payout 2024-04-01",
        "miscellaneous one-off transaction",
    )

    val clusterer = TemplateClusterer()
    val clusters = clusterer.cluster(texts = corpus)

    println("=== TemplateClusterer: ${corpus.size} texts → ${clusters.size} clusters ===")
    println()

    clusters.forEachIndexed { index, cluster ->
        println("Cluster ${index + 1}  [${cluster.members.size} member(s)]")
        println("  signature : ${cluster.signature.render()}")
        cluster.members.forEach { member ->
            println("  member    : $member")
        }
        println()
    }

    println("=== Sizes (largest first) ===")
    clusters.forEach { cluster ->
        val sig = cluster.signature.render()
        println("  ${"%-60s".format(sig)} → ${cluster.members.size}")
    }
}
