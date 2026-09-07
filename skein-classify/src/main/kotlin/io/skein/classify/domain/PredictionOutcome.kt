package io.skein.classify.domain

/**
 * One evaluated case: the ground-truth [expected] label and what the model actually said.
 *
 * This is the unit [EvaluationReportFactory] consumes. It carries a full [Prediction] — including
 * the ranked [Prediction.alternatives] — because top-k accuracy, log loss, the Brier score and the
 * calibration bins all need the whole probability distribution, not just the winning label.
 *
 * Calibration fitting needs the *raw, pre-softmax* scores instead; that hand-off type is
 * [CalibrationSample].
 */
data class PredictionOutcome(val expected: Label, val prediction: Prediction)
