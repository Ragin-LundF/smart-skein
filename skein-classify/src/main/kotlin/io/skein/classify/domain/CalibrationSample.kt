package io.skein.classify.domain

/**
 * One held-out observation scored by a model: the [trueLabel] and the model's **raw** per-label
 * [logScores], before any softmax or calibration.
 *
 * Calibration fitting needs the pre-softmax scores, which is why this is not a
 * [PredictionOutcome] — that type carries a finished [Prediction] and serves the metric math in
 * [EvaluationReportFactory]. The two are deliberately separate: one measures a model, the other
 * adjusts it.
 *
 * Only the differences between scores matter, since the softmax is shift-invariant, so an
 * implementation may return scores carrying any common offset.
 */
data class CalibrationSample(val trueLabel: Label, val logScores: Map<Label, Double>)
