package io.skein.classify.domain

/**
 * A record's feature text with every label observed for it — training data as it exists *before*
 * featurisation.
 *
 * This is what cross-validation splits, rather than [MultiLabeledFeatures], and the distinction is
 * the difference between an honest score and an inflated one. A vectorizer that fits anything to
 * the corpus — a vocabulary, a document-frequency table — must be fitted on each fold's training
 * rows alone. Splitting after featurisation makes that impossible: the statistics of the held-out
 * rows are already baked into every feature.
 *
 * [group] names the origin this row shares with its siblings — see [MultiLabeledFeatures.group].
 */
data class MultiLabeledText(
    val featureText: String,
    val labels: Set<Label>,
    val group: String? = null,
)
