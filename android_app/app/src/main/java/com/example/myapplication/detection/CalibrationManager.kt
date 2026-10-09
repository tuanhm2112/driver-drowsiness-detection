package com.example.myapplication.detection

import kotlin.math.sqrt
import kotlin.math.max

class CalibrationManager(
    private val calibrationSeconds: Float = LandmarkConstants.CALIBRATION_SECONDS
) {
    private val calibrationBuffer = mutableListOf<FloatArray>()
    private var startTime: Long = 0L
    
    // Custom floors for each feature
    // 0: EAR, 1: MAR, 2: BrowEye, 3: GazeOffset, 4: HeadRatio, 5: NoseMidX, 6: NoseMidY
    private val varianceFloors = floatArrayOf(
        0.005f, 0.030f, 0.002f, 0.005f, 0.020f, 0.020f, 0.020f
    )
    
    var baselineMean: FloatArray? = null
        private set
        
    var baselineStd: FloatArray? = null
        private set

    val isCalibrated: Boolean
        get() = baselineMean != null

    fun start() {
        startTime = System.currentTimeMillis()
        calibrationBuffer.clear()
        baselineMean = null
        baselineStd = null
    }

    /**
     * @return Thời gian còn lại (giây) để hoàn thành calibration
     */
    fun addSample(features: FloatArray): Float {
        if (startTime == 0L) start()
        
        val elapsed = (System.currentTimeMillis() - startTime) / 1000f
        calibrationBuffer.add(features)

        if (elapsed >= calibrationSeconds && calibrationBuffer.isNotEmpty()) {
            finalizeCalibration()
        }
        return max(0f, calibrationSeconds - elapsed)
    }

    /**
     * Personalize features: (features - baselineMean) / baselineStd
     */
    fun personalize(features: FloatArray): FloatArray {
        val mean = baselineMean ?: return features
        val std = baselineStd ?: return features
        
        return FloatArray(features.size) { i ->
            (features[i] - mean[i]) / std[i]
        }
    }

    private fun finalizeCalibration() {
        if (calibrationBuffer.isEmpty()) return
        
        val n = calibrationBuffer.size
        val dim = calibrationBuffer[0].size
        val mean = FloatArray(dim)
        val std = FloatArray(dim)
        
        // Tính Mean
        for (sample in calibrationBuffer) {
            for (i in 0 until dim) {
                mean[i] += sample[i]
            }
        }
        for (i in 0 until dim) {
            mean[i] /= n.toFloat()
        }
        
        // Tính Std
        for (sample in calibrationBuffer) {
            for (i in 0 until dim) {
                val diff = sample[i] - mean[i]
                std[i] += diff * diff
            }
        }
        for (i in 0 until dim) {
            var s = sqrt(std[i] / n.toFloat())
            val floor = if (i < varianceFloors.size) varianceFloors[i] else 0.005f
            
            if (s < 1e-4f) {
                s = 1.0f // Fallback to raw difference if variance is practically zero
            } else if (s < floor) {
                s = floor // Custom floor based on feature type
            }
            std[i] = s
        }
        
        baselineMean = mean
        baselineStd = std
    }
}
