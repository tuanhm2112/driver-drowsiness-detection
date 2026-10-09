package com.example.myapplication.detection

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.nio.FloatBuffer
import kotlin.math.exp

data class DetectionResult(
    val isDrowsy: Boolean,
    val drowsyProbability: Float,
    val alertProbability: Float
)

interface DrowsinessPredictor {
    fun predict(windowData: Array<FloatArray>): DetectionResult
    fun close()
}

class DrowsinessClassifier(
    modelPath: String,
    private val scalerMean: FloatArray,
    private val scalerStd: FloatArray,
    private val threshold: Float = LandmarkConstants.DROWSY_THRESHOLD
) : DrowsinessPredictor {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val session: OrtSession = env.createSession(modelPath)

    override fun predict(windowData: Array<FloatArray>): DetectionResult {
        // windowData: [60, 7]
        val sequenceLength = windowData.size
        val featureDim = windowData[0].size
        
        // 1. Normalize global (với scalerMean, scalerStd)
        // 2. Flatten thành mảng 1 chiều [60 * 7] để nạp vào tensor
        val flatInput = FloatArray(sequenceLength * featureDim)
        
        for (t in 0 until sequenceLength) {
            for (f in 0 until featureDim) {
                val normalizedValue = (windowData[t][f] - scalerMean[f]) / scalerStd[f]
                flatInput[t * featureDim + f] = normalizedValue
            }
        }

        // Tạo tensor với shape [1, 60, 7] (batch_size=1)
        val inputTensor = OnnxTensor.createTensor(
            env,
            FloatBuffer.wrap(flatInput),
            longArrayOf(1, sequenceLength.toLong(), featureDim.toLong())
        )

        // Chạy inference
        val output = session.run(mapOf("input" to inputTensor))
        
        // Lấy logits: output[0] là kết quả của model (shape [1, 2])
        @Suppress("UNCHECKED_CAST")
        val logits = (output[0].value as Array<FloatArray>)[0]
        
        // Áp dụng Softmax để lấy xác suất
        val maxLogit = maxOf(logits[0], logits[1])
        val exp0 = exp(logits[0] - maxLogit)
        val exp1 = exp(logits[1] - maxLogit)
        val sumExp = exp0 + exp1
        
        val alertProb = (exp0 / sumExp).toFloat()
        val drowsyProb = (exp1 / sumExp).toFloat()

        // Giải phóng tensor
        inputTensor.close()
        output.close()

        return DetectionResult(
            isDrowsy = drowsyProb >= threshold,
            drowsyProbability = drowsyProb,
            alertProbability = alertProb
        )
    }

    override fun close() {
        session.close()
        env.close()
    }
}
