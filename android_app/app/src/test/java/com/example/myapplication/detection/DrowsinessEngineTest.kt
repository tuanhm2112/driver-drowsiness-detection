package com.example.myapplication.detection

import org.junit.Assert.assertEquals
import org.junit.Test

class DrowsinessEngineTest {
    @Test
    fun staysCalibratingUntilNextSampleAfterImmediateCalibration() {
        val engine = createEngine()

        val first = engine.processFeatures(sampleFeatures(), 0L)
        val second = engine.processFeatures(sampleFeatures(), 100L)

        assertEquals(DetectionState.CALIBRATING, first.state)
        assertEquals(DetectionState.BUFFERING, second.state)
        engine.close()
    }

    @Test
    fun becomesMonitoringWhenBufferIsFullAndPredictionIsAlert() {
        val engine = createEngine(
            predictor = FakePredictor(
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f)
            )
        )

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)
        val snapshot = engine.processFeatures(sampleFeatures(), 200L)

        assertEquals(DetectionState.MONITORING, snapshot.state)
        engine.close()
    }

    @Test
    fun escalatesFromWarningToAlertAfterDrowsyConfirmationDuration() {
        val engine = createEngine(
            predictor = FakePredictor(
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f)
            )
        )

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)
        val warning = engine.processFeatures(sampleFeatures(), 200L)
        val stillWarning = engine.processFeatures(sampleFeatures(), 1300L)
        val alert = engine.processFeatures(sampleFeatures(), 2200L)

        assertEquals(DetectionState.WARNING, warning.state)
        assertEquals(DetectionState.WARNING, stillWarning.state)
        assertEquals(DetectionState.ALERT, alert.state)
        engine.close()
    }

    @Test
    fun resetsToFaceLostAfterMissingFaceTimeout() {
        val engine = createEngine(
            predictor = FakePredictor(
                DetectionResult(isDrowsy = false, drowsyProbability = 0.10f, alertProbability = 0.90f)
            )
        )

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)
        engine.processFeatures(sampleFeatures(), 200L)

        val temporaryLoss = engine.processNoFace(900L)
        val faceLost = engine.processNoFace(1300L)

        assertEquals(DetectionState.WARNING, temporaryLoss.state)
        assertEquals(DetectionState.FACE_LOST, faceLost.state)
        engine.close()
    }

    @Test
    fun scoreDecaysFromAlertToWarningBeforeMonitoring() {
        val predictor = SequencePredictor(
            listOf(
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f)
            )
        )
        val engine = createEngine(predictor = predictor)

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)
        val warning = engine.processFeatures(sampleFeatures(), 200L)
        val stillWarning = engine.processFeatures(sampleFeatures(), 1300L)
        val alert = engine.processFeatures(sampleFeatures(), 2300L)
        val recoveryWarning = engine.processFeatures(sampleFeatures(), 2800L)
        val monitoring = engine.processFeatures(sampleFeatures(), 4400L)

        assertEquals(DetectionState.WARNING, warning.state)
        assertEquals(DetectionState.WARNING, stillWarning.state)
        assertEquals(DetectionState.ALERT, alert.state)
        assertEquals(DetectionState.WARNING, recoveryWarning.state)
        assertEquals(DetectionState.MONITORING, monitoring.state)
        engine.close()
    }

    @Test
    fun repeatedWarningTransitionsTriggerAlertAndResetAccumulation() {
        val predictor = SequencePredictor(
            listOf(
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f)
            )
        )
        val engine = createEngine(predictor = predictor)

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)

        val firstWarning = engine.processFeatures(sampleFeatures(), 200L)
        val firstRecovery = engine.processFeatures(sampleFeatures(), 1_000L)
        engine.processFeatures(sampleFeatures(), 11_900L)
        val secondWarning = engine.processFeatures(sampleFeatures(), 12_000L)
        val secondRecovery = engine.processFeatures(sampleFeatures(), 12_800L)
        engine.processFeatures(sampleFeatures(), 23_900L)
        val accumulatedAlert = engine.processFeatures(sampleFeatures(), 24_000L)
        val recoveryAfterAlert = engine.processFeatures(sampleFeatures(), 24_800L)
        engine.processFeatures(sampleFeatures(), 35_900L)
        val warningAfterReset = engine.processFeatures(sampleFeatures(), 36_000L)

        assertEquals(DetectionState.WARNING, firstWarning.state)
        assertEquals(DetectionState.MONITORING, firstRecovery.state)
        assertEquals(DetectionState.WARNING, secondWarning.state)
        assertEquals(DetectionState.MONITORING, secondRecovery.state)
        assertEquals(DetectionState.ALERT, accumulatedAlert.state)
        assertEquals(DetectionState.MONITORING, recoveryAfterAlert.state)
        assertEquals(DetectionState.WARNING, warningAfterReset.state)
        engine.close()
    }

    @Test
    fun warningTransitionCooldownPreventsRapidDuplicateEvents() {
        val predictor = SequencePredictor(
            listOf(
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = false, drowsyProbability = 0.20f, alertProbability = 0.80f),
                DetectionResult(isDrowsy = true, drowsyProbability = 0.85f, alertProbability = 0.15f)
            )
        )
        val engine = createEngine(predictor = predictor)

        engine.processFeatures(sampleFeatures(), 0L)
        engine.processFeatures(sampleFeatures(), 100L)

        val firstWarning = engine.processFeatures(sampleFeatures(), 200L)
        val quickRecovery = engine.processFeatures(sampleFeatures(), 400L)
        val quickWarning = engine.processFeatures(sampleFeatures(), 500L)
        val recovery = engine.processFeatures(sampleFeatures(), 700L)
        engine.processFeatures(sampleFeatures(), 11_900L)
        val secondCountedWarning = engine.processFeatures(sampleFeatures(), 12_000L)
        val secondRecovery = engine.processFeatures(sampleFeatures(), 12_800L)
        engine.processFeatures(sampleFeatures(), 23_900L)
        val thirdCountedWarning = engine.processFeatures(sampleFeatures(), 24_000L)

        assertEquals(DetectionState.WARNING, firstWarning.state)
        assertEquals(DetectionState.MONITORING, quickRecovery.state)
        assertEquals(DetectionState.WARNING, quickWarning.state)
        assertEquals(DetectionState.MONITORING, recovery.state)
        assertEquals(DetectionState.WARNING, secondCountedWarning.state)
        assertEquals(DetectionState.MONITORING, secondRecovery.state)
        assertEquals(DetectionState.ALERT, thirdCountedWarning.state)
        engine.close()
    }

    private fun createEngine(
        predictor: DrowsinessPredictor = FakePredictor(
            DetectionResult(isDrowsy = false, drowsyProbability = 0.10f, alertProbability = 0.90f)
        )
    ): DrowsinessEngine {
        return DrowsinessEngine(
            slidingWindow = SlidingWindowBuffer(maxSize = 2),
            calibrationManager = CalibrationManager(calibrationSeconds = 0f),
            drowsinessClassifier = predictor
        )
    }

    private fun sampleFeatures(): FloatArray {
        return floatArrayOf(0.3f, 0.2f, 0.1f, 0.05f, 1.0f, 0.0f, 0.0f)
    }
}

private class FakePredictor(
    private val result: DetectionResult
) : DrowsinessPredictor {
    override fun predict(windowData: Array<FloatArray>): DetectionResult = result

    override fun close() = Unit
}

private class SequencePredictor(
    private val results: List<DetectionResult>
) : DrowsinessPredictor {
    private var index = 0

    override fun predict(windowData: Array<FloatArray>): DetectionResult {
        val result = results.getOrElse(index) { results.last() }
        index++
        return result
    }

    override fun close() = Unit
}
