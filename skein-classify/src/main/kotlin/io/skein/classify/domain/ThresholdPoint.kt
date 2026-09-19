package io.skein.classify.domain

/**
 * One row of a threshold sweep: what the model scores when labels are accepted at [threshold].
 *
 * [precision], [recall] and [f1] are the micro-averaged figures, and [coverage] is the fraction of
 * cases that received at least one label at all. Coverage is the column most often left out and the
 * one that makes the trade-off legible — a threshold with excellent precision that labels a third
 * of the corpus is a different product decision from one that labels all of it.
 */
data class ThresholdPoint(
    val threshold: Double,
    val precision: Double,
    val recall: Double,
    val f1: Double,
    val coverage: Double,
)
