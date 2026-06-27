package io.skein.cli

/** Which classifier a CLI run trains. Persisted in the model file so a resumed run rebuilds the same kind. */
enum class ClassifierKindEnum {
    NAIVE_BAYES,
    LOGISTIC_REGRESSION,
}
