package io.skein.classify.application

/** Which classifier a model was trained with. Persisted in the model file so a loaded model rebuilds the same kind. */
enum class ClassifierKindEnum {
    NAIVE_BAYES,
    LOGISTIC_REGRESSION,
}
