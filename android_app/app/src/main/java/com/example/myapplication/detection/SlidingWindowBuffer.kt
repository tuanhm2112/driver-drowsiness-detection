package com.example.myapplication.detection

class SlidingWindowBuffer(private val maxSize: Int = LandmarkConstants.SEQUENCE_LENGTH) {
    private val buffer = ArrayDeque<FloatArray>(maxSize)

    fun push(features: FloatArray) {
        if (buffer.size >= maxSize) {
            buffer.removeFirst()
        }
        buffer.addLast(features)
    }

    fun isFull(): Boolean = buffer.size >= maxSize

    fun size(): Int = buffer.size

    fun capacity(): Int = maxSize

    fun toTensor(): Array<FloatArray> = buffer.toTypedArray()

    fun clear() {
        buffer.clear()
    }
}
