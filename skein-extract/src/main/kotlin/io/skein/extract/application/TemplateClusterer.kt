package io.skein.extract.application

import io.skein.extract.domain.TemplateCluster
import io.skein.text.application.TypedTokenizer
import io.skein.text.domain.PatternSignature

/**
 * Unsupervised discovery of recurring layouts: groups texts by their [PatternSignature] (the ordered
 * sequence of token types). Texts that share a structural shape land in the same cluster, which is
 * the starting point for wrapper-induction-style extraction.
 */
class TemplateClusterer(private val tokenizer: TypedTokenizer = TypedTokenizer()) {

    /** Clusters [texts] by structural signature, largest cluster first. */
    fun cluster(texts: List<String>): List<TemplateCluster> {
        val membersBySignature = LinkedHashMap<PatternSignature, MutableList<String>>()
        for (text in texts) {
            val signature = PatternSignature.of(tokens = tokenizer.tokenize(text = text))
            membersBySignature.getOrPut(key = signature) { ArrayList() }.add(text)
        }
        return membersBySignature.entries
            .map { entry -> TemplateCluster(signature = entry.key, members = entry.value.toList()) }
            .sortedByDescending { cluster -> cluster.members.size }
    }
}
