package io.skein.classify.application

/** Which classifier a model was trained with. Persisted in the model file so a loaded model rebuilds the same kind. */
enum class ClassifierKindEnum {
    NAIVE_BAYES,
    LOGISTIC_REGRESSION,

    /**
     * One-vs-rest logistic heads fitted by L-BFGS — a model that can assign several labels to a
     * record, or none. Persisted as fitted weights rather than as replayable observations, because
     * a batch-fitted model has no incremental history to replay.
     *
     * Appended, never reordered: the ordinal is what a `.skein` file stores.
     */
    MULTI_LABEL_LOGISTIC,
}
