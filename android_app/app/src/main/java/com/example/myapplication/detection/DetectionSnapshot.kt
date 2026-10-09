package com.example.myapplication.detection

enum class DetectionState {
    CALIBRATING,
    BUFFERING,
    MONITORING,
    WARNING,
    ALERT,
    FACE_LOST
}

data class DetectionSnapshot(
    val state: DetectionState,
    val statusText: String,
    val featuresText: String = "EAR: -- | MAR: --",
    val probabilitiesText: String = "Alert: 0% | Drowsy: 0%"
)
