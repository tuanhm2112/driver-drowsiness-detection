package com.example.myapplication.detection

import android.content.Context
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarkerResult
import org.json.JSONArray
import java.io.File
import java.io.InputStream
import java.util.ArrayDeque

class DrowsinessEngine internal constructor(
    private val featureCalculator: FeatureCalculator = FeatureCalculator(),
    private val slidingWindow: SlidingWindowBuffer = SlidingWindowBuffer(),
    private val calibrationManager: CalibrationManager = CalibrationManager(),
    private val featureSmoother: FeatureSmoother = FeatureSmoother(),
    private val drowsinessClassifier: DrowsinessPredictor
) {
    private var currentState = DetectionState.CALIBRATING
    private var lastFaceSeenAtMs: Long? = null
    private var lastPredictionAtMs: Long? = null
    private var drowsinessScoreMs = 0f
    private var lastValidFeatures: FloatArray? = null
    private var lastFeaturesText = "EAR: -- | MAR: --"
    private val warningEvents = ArrayDeque<Long>()

    fun processResult(result: FaceLandmarkerResult): DetectionSnapshot {
        val timestampMs = result.timestampMs()
        if (result.faceLandmarks().isEmpty()) {
            return processNoFace(timestampMs)
        }

        lastFaceSeenAtMs = timestampMs

        val face = result.faceLandmarks()[0]
        val rawLandmarks = Array(70) { FloatArray(3) }
        for (i in 0 until 70) {
            val mpIndex = LandmarkConstants.KEY_LANDMARKS[i]
            val point = face[mpIndex]
            rawLandmarks[i][0] = point.x()
            rawLandmarks[i][1] = point.y()
            rawLandmarks[i][2] = point.z()
        }

        val normalizedLm = featureCalculator.normalizeLandmarks(rawLandmarks)
        val rawFeatures = featureCalculator.computeFeatures(normalizedLm)
        return processFeatures(rawFeatures, timestampMs)
    }

    fun close() {
        drowsinessClassifier.close()
    }

    internal fun processFeatures(rawFeatures: FloatArray, timestampMs: Long): DetectionSnapshot {
        lastFaceSeenAtMs = timestampMs

        val featuresText = formatFeatures(rawFeatures)
        lastFeaturesText = featuresText

        if (!calibrationManager.isCalibrated) {
            val timeLeft = calibrationManager.addSample(rawFeatures)
            return snapshot(
                state = DetectionState.CALIBRATING,
                statusText = String.format("Giữ tư thế tự nhiên (%.1f s)", timeLeft),
                featuresText = featuresText
            )
        }

        val personalized = calibrationManager.personalize(rawFeatures)
        val smoothed = featureSmoother.smooth(personalized)
        lastValidFeatures = smoothed
        slidingWindow.push(smoothed)

        return checkAndPredict(timestampMs, featuresText)
    }

    internal fun processNoFace(timestampMs: Long): DetectionSnapshot {
        if (!calibrationManager.isCalibrated) {
            return snapshot(
                state = DetectionState.CALIBRATING,
                statusText = "Đưa khuôn mặt vào khung hình để hiệu chuẩn..."
            )
        }

        val faceLostDurationMs = lastFaceSeenAtMs?.let { timestampMs - it } ?: Long.MAX_VALUE
        if (faceLostDurationMs >= LandmarkConstants.FACE_LOST_RESET_MS) {
            slidingWindow.clear()
            featureSmoother.reset()
            lastValidFeatures = null
            lastPredictionAtMs = null
            drowsinessScoreMs = 0f
            clearWarningEvents()

            return snapshot(
                state = DetectionState.FACE_LOST,
                statusText = "Không thấy rõ khuôn mặt"
            )
        }

        return snapshot(
            state = DetectionState.WARNING,
            statusText = "Đang tìm lại khuôn mặt",
            featuresText = lastFeaturesText
        )
    }

    private fun checkAndPredict(
        timestampMs: Long,
        featuresText: String = "EAR: -- | MAR: --"
    ): DetectionSnapshot {
        if (!slidingWindow.isFull()) {
            lastPredictionAtMs = null
            drowsinessScoreMs = 0f
            return snapshot(
                state = DetectionState.BUFFERING,
                statusText = "Đang làm quen với khuôn mặt (${slidingWindow.size()}/${slidingWindow.capacity()})",
                featuresText = featuresText
            )
        }

        val pred = drowsinessClassifier.predict(slidingWindow.toTensor())
        val deltaMs = predictionDeltaMs(timestampMs)
        updateDrowsinessScore(pred, deltaMs)

        val probabilitiesText = String.format(
            "Alert: %.1f%% | Drowsy: %.1f%%",
            pred.alertProbability * 100,
            pred.drowsyProbability * 100
        )

        if (drowsinessScoreMs >= LandmarkConstants.DROWSY_SCORE_ALERT_MS) {
            clearWarningEvents()
            return snapshot(
                state = DetectionState.ALERT,
                statusText = "Phát hiện dấu hiệu nguy hiểm",
                featuresText = featuresText,
                probabilitiesText = probabilitiesText
            )
        }

        if (pred.isDrowsy || drowsinessScoreMs >= LandmarkConstants.DROWSY_SCORE_WARNING_MS) {
            recordWarningTransition(timestampMs)
            if (hasAccumulatedWarningEvents(timestampMs)) {
                clearWarningEvents()
                return snapshot(
                    state = DetectionState.ALERT,
                    statusText = "Dấu hiệu mệt mỏi lặp lại",
                    featuresText = featuresText,
                    probabilitiesText = probabilitiesText
                )
            }

            return snapshot(
                state = DetectionState.WARNING,
                statusText = "Có dấu hiệu cần chú ý",
                featuresText = featuresText,
                probabilitiesText = probabilitiesText
            )
        }

        pruneWarningEvents(timestampMs)
        return snapshot(
            state = DetectionState.MONITORING,
            statusText = "Đang theo dõi ổn định",
            featuresText = featuresText,
            probabilitiesText = probabilitiesText
        )
    }

    private fun predictionDeltaMs(timestampMs: Long): Long {
        val deltaMs = lastPredictionAtMs?.let { (timestampMs - it).coerceAtLeast(0L) } ?: 0L
        lastPredictionAtMs = timestampMs
        return deltaMs
    }

    private fun updateDrowsinessScore(prediction: DetectionResult, deltaMs: Long) {
        if (prediction.isDrowsy) {
            drowsinessScoreMs += deltaMs.toFloat()
        } else {
            drowsinessScoreMs -= deltaMs * LandmarkConstants.DROWSY_SCORE_DECAY_MULTIPLIER
        }

        drowsinessScoreMs = drowsinessScoreMs.coerceIn(0f, LandmarkConstants.DROWSY_SCORE_ALERT_MS * 2f)
    }

    private fun recordWarningTransition(timestampMs: Long) {
        pruneWarningEvents(timestampMs)
        if (currentState == DetectionState.WARNING) return

        val lastWarningEventAtMs = warningEvents.peekLast()
        if (lastWarningEventAtMs != null && timestampMs - lastWarningEventAtMs < LandmarkConstants.WARNING_EVENT_COOLDOWN_MS) {
            return
        }

        warningEvents.addLast(timestampMs)
    }

    private fun hasAccumulatedWarningEvents(timestampMs: Long): Boolean {
        pruneWarningEvents(timestampMs)
        return warningEvents.size >= LandmarkConstants.WARNING_EVENT_COUNT_ALERT
    }

    private fun pruneWarningEvents(timestampMs: Long) {
        while (warningEvents.isNotEmpty() && timestampMs - warningEvents.first() > LandmarkConstants.WARNING_EVENT_WINDOW_MS) {
            warningEvents.removeFirst()
        }
    }

    private fun clearWarningEvents() {
        warningEvents.clear()
    }

    private fun snapshot(
        state: DetectionState,
        statusText: String,
        featuresText: String = "EAR: -- | MAR: --",
        probabilitiesText: String = "Alert: 0% | Drowsy: 0%"
    ): DetectionSnapshot {
        currentState = state
        return DetectionSnapshot(
            state = state,
            statusText = statusText,
            featuresText = featuresText,
            probabilitiesText = probabilitiesText
        )
    }

    private fun formatFeatures(features: FloatArray): String {
        return String.format("EAR: %.2f | MAR: %.2f", features[0], features[1])
    }

    companion object {
        fun create(context: Context): DrowsinessEngine {
            val modelFile = File(context.filesDir, "drowsy_model.onnx")
            val dataFile = File(context.filesDir, "drowsy_model.onnx.data")

            if (!modelFile.exists()) {
                context.assets.open("drowsy_model.onnx").use { input ->
                    modelFile.outputStream().use { output -> input.copyTo(output) }
                }
            }
            if (!dataFile.exists()) {
                context.assets.open("drowsy_model.onnx.data").use { input ->
                    dataFile.outputStream().use { output -> input.copyTo(output) }
                }
            }

            val meanJson = readJsonArray(context.assets.open("scaler_mean.json"))
            val stdJson = readJsonArray(context.assets.open("scaler_std.json"))

            val meanArray = FloatArray(meanJson.length()) { i -> meanJson.getDouble(i).toFloat() }
            val stdArray = FloatArray(stdJson.length()) { i -> stdJson.getDouble(i).toFloat() }

            val classifier = DrowsinessClassifier(modelFile.absolutePath, meanArray, stdArray)
            return DrowsinessEngine(drowsinessClassifier = classifier)
        }

        private fun readJsonArray(inputStream: InputStream): JSONArray {
            val size = inputStream.available()
            val buffer = ByteArray(size)
            inputStream.read(buffer)
            inputStream.close()
            return JSONArray(String(buffer, Charsets.UTF_8))
        }
    }
}
