package io.skein.classify.domain

/**
 * A feature vector paired with **every** label observed for it — the unit of multi-label training
 * data, and the multi-label sibling of [LabeledFeatures].
 *
 * [labels] may be empty, and that is meaningful rather than missing: an explicit negative says
 * "none of the known labels apply here", which is exactly what teaches a one-vs-rest head where its
 * label does *not* belong. A corpus of only positives cannot express that.
 *
 * [group] names where this observation came from — typically the rule, template or document that
 * generated it. [io.skein.classify.application.GroupedSplitter] keeps a group whole inside one
 * fold, so siblings cannot sit on both sides of a split and inflate the score. `null` means
 * ungrouped, which a grouped split treats as a group of one.
 */
data class MultiLabeledFeatures(
    val features: FeatureVector,
    val labels: Set<Label>,
    val group: String? = null,
)
