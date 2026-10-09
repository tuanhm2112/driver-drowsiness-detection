package com.example.myapplication

import android.media.AudioManager
import android.media.ToneGenerator
import android.os.SystemClock

class AlertManager {
    private val toneGenerator = ToneGenerator(AudioManager.STREAM_ALARM, 100)
    private var lastBeepTime = 0L

    fun triggerAlarm() {
        val now = SystemClock.elapsedRealtime()
        if (now - lastBeepTime > 1000) {
            toneGenerator.startTone(ToneGenerator.TONE_CDMA_EMERGENCY_RINGBACK, 500)
            lastBeepTime = now
        }
    }

    fun stopAll() {
        toneGenerator.stopTone()
    }

    fun release() {
        stopAll()
        toneGenerator.release()
    }
}
