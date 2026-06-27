package io.skein.text.domain

/**
 * Self-learned word frequency model used to recognize real words during broken-word repair.
 *
 * Privacy by design: a word is only considered *known* once it has been seen at least
 * [minKeepFrequency] times. Rare words — which are far more likely to be personal data
 * (names, IBANs, customer numbers) — never cross the threshold and are never treated as
 * vocabulary. [minKeepFrequency] is the calibration knob: raise it in production for
 * stronger privacy, keep it low for small training corpora.
 *
 * [serialize]/[deserialize] persist only the above-threshold vocabulary, so sub-threshold
 * (potential-PII) words are never written to disk.
 */
class FrequencyModel(private val minKeepFrequency: Int = DEFAULT_MIN_KEEP_FREQUENCY) {

    private val counts = HashMap<String, Int>()

    /** Records one occurrence of [word] (case-insensitive). Blank input is ignored. */
    fun learn(word: String) {
        if (word.isBlank()) {
            return
        }
        val key = word.lowercase()
        counts[key] = (counts[key] ?: 0) + 1
    }

    /** Records every word in [words]. */
    fun learnAll(words: Iterable<String>) {
        words.forEach { word -> learn(word) }
    }

    /** Observed count of [word], or `0` when it is below the privacy threshold or unseen. */
    fun frequency(word: String): Int {
        val count = counts[word.lowercase()] ?: 0
        return if (count >= minKeepFrequency) count else 0
    }

    /** Whether [word] has been seen often enough to count as real vocabulary. */
    fun isKnown(word: String): Boolean {
        return frequency(word) > 0
    }

    /** Vocabulary retained after the privacy threshold; rare (potential-PII) words are excluded. */
    fun knownWords(): Set<String> {
        return counts.filterValues { count -> count >= minKeepFrequency }.keys
    }

    /**
     * Serializes the model to text: the threshold on the first line, then one `count<TAB>word` line
     * per known word. Only above-threshold words are written — rare (potential-PII) words are dropped.
     * Words must not contain tab or newline characters.
     */
    fun serialize(): String {
        val builder = StringBuilder()
        builder.append(minKeepFrequency).append('\n')
        for (word in knownWords()) {
            builder.append(counts.getValue(word)).append('\t').append(word).append('\n')
        }
        return builder.toString()
    }

    companion object {
        const val DEFAULT_MIN_KEEP_FREQUENCY = 1

        /** Reconstructs a model from [serialize] output. */
        fun deserialize(serialized: String): FrequencyModel {
            val lines = serialized.lineSequence().filter { line -> line.isNotBlank() }.toList()
            require(value = lines.isNotEmpty()) { "serialized frequency model is empty" }
            val model = FrequencyModel(minKeepFrequency = lines.first().trim().toInt())
            for (line in lines.drop(n = 1)) {
                val parts = line.split('\t', limit = 2)
                model.counts[parts[1]] = parts[0].trim().toInt()
            }
            return model
        }
    }
}
