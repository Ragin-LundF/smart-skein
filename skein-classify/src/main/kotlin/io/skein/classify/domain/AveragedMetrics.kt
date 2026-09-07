package io.skein.classify.domain

/** Precision, recall and F1 collapsed to one number each under a single averaging convention. */
data class AveragedMetrics(val precision: Double, val recall: Double, val f1: Double)
