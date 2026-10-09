package com.example.myapplication.detection

import kotlin.math.sqrt

object LandmarkConstants {
    // 70 điểm landmark quan trọng (khớp với Python)
    val KEY_LANDMARKS = intArrayOf(
        33, 160, 158, 133, 153, 144,
        362, 385, 387, 263, 373, 380,
        70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
        300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
        61, 291, 13, 14, 82, 312, 87, 317, 78, 308, 95, 324, 0, 17,
        1, 2, 5, 4, 19, 94,
        468, 469, 470, 471, 472,
        473, 474, 475, 476, 477,
        10, 152, 234, 454, 127, 356, 162, 389
    )

    // Vị trí trong mảng 70 điểm (index 0-69)
    val LEFT_EYE = intArrayOf(0, 1, 2, 3, 4, 5)
    val RIGHT_EYE = intArrayOf(6, 7, 8, 9, 10, 11)
    val LEFT_BROW_CENTER = intArrayOf(12, 13, 14, 15, 16)
    val RIGHT_BROW_CENTER = intArrayOf(22, 23, 24, 25, 26)
    val MOUTH = intArrayOf(32, 33, 34, 35, 36, 37, 38, 39)
    val LEFT_IRIS = intArrayOf(52, 53, 54, 55, 56)
    val RIGHT_IRIS = intArrayOf(57, 58, 59, 60, 61)
    val NOSE = 44
    val CHIN = 63
    val FOREHEAD = 62
    val LEFT_FACE = 64
    val RIGHT_FACE = 65

    const val SEQUENCE_LENGTH = 60
    const val CALIBRATION_SECONDS = 5.0f
    const val DROWSY_THRESHOLD = 0.50f
    const val DROWSY_SCORE_WARNING_MS = 600f
    const val DROWSY_SCORE_ALERT_MS = 1800f
    const val DROWSY_SCORE_DECAY_MULTIPLIER = 0.75f
    const val FACE_LOST_RESET_MS = 1000L
    const val WARNING_EVENT_WINDOW_MS = 120_000L
    const val WARNING_EVENT_COOLDOWN_MS = 10_000L
    const val WARNING_EVENT_COUNT_ALERT = 3
}

class FeatureCalculator {

    /**
     * Normalize landmarks: center by mean, scale by face_size
     */
    fun normalizeLandmarks(landmarks: Array<FloatArray>): Array<FloatArray> {
        val normalized = landmarks.map { it.clone() }.toTypedArray()

        var meanX = 0f; var meanY = 0f
        for (lm in normalized) { meanX += lm[0]; meanY += lm[1] }
        meanX /= normalized.size
        meanY /= normalized.size
        for (lm in normalized) { lm[0] -= meanX; lm[1] -= meanY }

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        for (lm in landmarks) {
            minX = minOf(minX, lm[0]); maxX = maxOf(maxX, lm[0])
            minY = minOf(minY, lm[1]); maxY = maxOf(maxY, lm[1])
        }
        val scale = maxOf(maxX - minX, maxY - minY, 1e-6f)
        for (lm in normalized) {
            lm[0] /= scale; lm[1] /= scale
        }

        var meanZ = 0f
        for (lm in normalized) { meanZ += lm[2] }
        meanZ /= normalized.size
        for (lm in normalized) { lm[2] = (lm[2] - meanZ) / scale }

        return normalized
    }

    /**
     * Tính 7 handcrafted features từ 70 landmarks đã normalize
     * Trả về mảng 7 phần tử: [EAR, MAR, BrowEye, Gaze, HeadRatio, NoseX, NoseY]
     */
    fun computeFeatures(lm: Array<FloatArray>): FloatArray {
        val leftEar = computeEAR(lm, LandmarkConstants.LEFT_EYE)
        val rightEar = computeEAR(lm, LandmarkConstants.RIGHT_EYE)
        val ear = (leftEar + rightEar) / 2f

        val mar = computeMAR(lm)

        val leftBrowEye = distance2D(meanPoint(lm, LandmarkConstants.LEFT_BROW_CENTER), meanPoint(lm, LandmarkConstants.LEFT_EYE))
        val rightBrowEye = distance2D(meanPoint(lm, LandmarkConstants.RIGHT_BROW_CENTER), meanPoint(lm, LandmarkConstants.RIGHT_EYE))
        val browEye = (leftBrowEye + rightBrowEye) / 2f

        val leftGaze = distance2D(meanPoint(lm, LandmarkConstants.LEFT_IRIS), meanPoint(lm, LandmarkConstants.LEFT_EYE))
        val rightGaze = distance2D(meanPoint(lm, LandmarkConstants.RIGHT_IRIS), meanPoint(lm, LandmarkConstants.RIGHT_EYE))
        val gazeOffset = (leftGaze + rightGaze) / 2f

        val forehead = lm[LandmarkConstants.FOREHEAD]
        val chin = lm[LandmarkConstants.CHIN]
        val leftFace = lm[LandmarkConstants.LEFT_FACE]
        val rightFace = lm[LandmarkConstants.RIGHT_FACE]
        
        val faceHeight = maxOf(distance2D(forehead, chin), 1e-6f)
        val faceWidth = maxOf(distance2D(leftFace, rightFace), 1e-6f)
        val headRatio = faceWidth / faceHeight

        val nose = lm[LandmarkConstants.NOSE]
        val noseMidX = nose[0] - (leftFace[0] + rightFace[0]) / 2f
        val noseMidY = (forehead[1] + chin[1]) / 2f - nose[1]

        return floatArrayOf(ear, mar, browEye, gazeOffset, headRatio, noseMidX, noseMidY)
    }

    private fun computeEAR(lm: Array<FloatArray>, eye: IntArray): Float {
        val vertical = distance2D(lm[eye[1]], lm[eye[5]]) + distance2D(lm[eye[2]], lm[eye[4]])
        val horizontal = 2f * maxOf(distance2D(lm[eye[0]], lm[eye[3]]), 1e-6f)
        return vertical / horizontal
    }

    private fun computeMAR(lm: Array<FloatArray>): Float {
        val m = LandmarkConstants.MOUTH
        val vertical = distance2D(lm[m[2]], lm[m[3]]) + distance2D(lm[m[4]], lm[m[5]]) + distance2D(lm[m[6]], lm[m[7]])
        val horizontal = 3f * maxOf(distance2D(lm[m[0]], lm[m[1]]), 1e-6f)
        return vertical / horizontal
    }

    private fun distance2D(a: FloatArray, b: FloatArray): Float {
        val dx = a[0] - b[0]; val dy = a[1] - b[1]
        return sqrt(dx * dx + dy * dy)
    }

    private fun meanPoint(lm: Array<FloatArray>, indices: IntArray): FloatArray {
        var x = 0f; var y = 0f
        for (i in indices) { x += lm[i][0]; y += lm[i][1] }
        return floatArrayOf(x / indices.size, y / indices.size)
    }
}
