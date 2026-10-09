package com.example.myapplication.detection

class FeatureSmoother(private val alpha: Float = 0.7f) {
    private var previousFeatures: FloatArray? = null

    fun smooth(currentFeatures: FloatArray): FloatArray {
        if (previousFeatures == null) {
            previousFeatures = currentFeatures.clone()
            return currentFeatures
        }

        val prev = previousFeatures!!
        val smoothed = FloatArray(currentFeatures.size)
        
        for (i in currentFeatures.indices) {
            smoothed[i] = alpha * currentFeatures[i] + (1 - alpha) * prev[i]
        }
        
        previousFeatures = smoothed.clone()
        return smoothed
    }

    fun reset() {
        previousFeatures = null
    }
}
