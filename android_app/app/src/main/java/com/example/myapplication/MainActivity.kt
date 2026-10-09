package com.example.myapplication

import android.Manifest
import android.app.PictureInPictureParams
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Rational
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.myapplication.detection.*
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

class MainActivity : AppCompatActivity() {

    private companion object {
        const val FRAME_PROCESSING_INTERVAL_MS = 33L
    }

    private enum class OnboardingStep {
        DEVICE_PLACEMENT,
        CALIBRATION,
        DONE
    }

    private lateinit var viewFinder: PreviewView
    private lateinit var statusCard: View
    private lateinit var controlsCard: View
    private lateinit var tvAlertBanner: TextView
    private lateinit var tvStatus: TextView
    private lateinit var tvGuidance: TextView
    private lateinit var progressStatus: ProgressBar
    private lateinit var btnMonitoringToggle: Button
    private lateinit var btnRecalibrate: Button
    private lateinit var onboardingOverlay: View
    private lateinit var ivOnboardingIllustration: ImageView
    private lateinit var tvOnboardingStep: TextView
    private lateinit var tvOnboardingTitle: TextView
    private lateinit var tvOnboardingBody: TextView
    private lateinit var btnOnboardingPrimary: Button

    private lateinit var cameraExecutor: ExecutorService
    private var cameraProvider: ProcessCameraProvider? = null
    private var faceLandmarker: FaceLandmarker? = null
    private lateinit var alertManager: AlertManager
    private var drowsinessEngine: DrowsinessEngine? = null
    private var isCameraBinding = false
    private var isCameraBound = false
    private var isMonitoringActive = false
    private var isMonitoringRequestedByUser = true
    private var isEnteringPictureInPicture = false
    private var currentOnboardingStep = OnboardingStep.DEVICE_PLACEMENT
    private var monitoringSessionToken = 0
    private var minAcceptedFrameTimestampMs = 0L
    private var lastProcessedFrameTimestampMs = 0L

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                startMonitoring(updateUserIntent = false)
            } else {
                Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_SHORT).show()
                showCameraPermissionRequiredUi()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setContentView(R.layout.activity_main)

        viewFinder = findViewById(R.id.viewFinder)
        statusCard = findViewById(R.id.statusCard)
        controlsCard = findViewById(R.id.controlsCard)
        tvAlertBanner = findViewById(R.id.tvAlertBanner)
        tvStatus = findViewById(R.id.tvStatus)
        tvGuidance = findViewById(R.id.tvGuidance)
        progressStatus = findViewById(R.id.progressStatus)
        btnMonitoringToggle = findViewById(R.id.btnMonitoringToggle)
        btnRecalibrate = findViewById(R.id.btnRecalibrate)
        onboardingOverlay = findViewById(R.id.onboardingOverlay)
        ivOnboardingIllustration = findViewById(R.id.ivOnboardingIllustration)
        tvOnboardingStep = findViewById(R.id.tvOnboardingStep)
        tvOnboardingTitle = findViewById(R.id.tvOnboardingTitle)
        tvOnboardingBody = findViewById(R.id.tvOnboardingBody)
        btnOnboardingPrimary = findViewById(R.id.btnOnboardingPrimary)

        cameraExecutor = Executors.newSingleThreadExecutor()
        alertManager = AlertManager()

        initEngine()
        setupMediaPipe()
        setupActions()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            setPictureInPictureParams(buildPipParams(autoEnterEnabled = true))
        }

        showOnboardingStep(OnboardingStep.DEVICE_PLACEMENT)
    }

    override fun onStart() {
        super.onStart()
        if (currentOnboardingStep == OnboardingStep.DONE && isMonitoringRequestedByUser && hasCameraPermission()) {
            startMonitoring(updateUserIntent = false)
        }
    }

    override fun onResume() {
        super.onResume()
        if (currentOnboardingStep != OnboardingStep.DONE) return

        if (isMonitoringRequestedByUser && hasCameraPermission()) {
            startMonitoring(updateUserIntent = false)
        } else {
            stopCamera()
            if (!hasCameraPermission()) {
                showCameraPermissionRequiredUi()
            } else {
                showMonitoringPausedUi()
            }
        }
    }

    override fun onStop() {
        super.onStop()
        if (!isInPictureInPictureMode && !isEnteringPictureInPicture) {
            stopCamera()
        }
    }

    override fun onUserLeaveHint() {
        super.onUserLeaveHint()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O &&
            !isInPictureInPictureMode &&
            hasCameraPermission() &&
            isCameraBound
        ) {
            enterMiniWindowMode()
        }
    }

    override fun onPictureInPictureModeChanged(
        isInPictureInPictureMode: Boolean,
        newConfig: Configuration
    ) {
        super.onPictureInPictureModeChanged(isInPictureInPictureMode, newConfig)
        isEnteringPictureInPicture = false
        applyPictureInPictureUi(isInPictureInPictureMode)
    }

    private fun enterMiniWindowMode() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return

        try {
            isEnteringPictureInPicture = true
            enterPictureInPictureMode(buildPipParams(autoEnterEnabled = false))
        } catch (e: IllegalStateException) {
            isEnteringPictureInPicture = false
            Log.w("MainActivity", "Unable to enter PiP", e)
        }
    }

    private fun buildPipParams(autoEnterEnabled: Boolean): PictureInPictureParams {
        val builder = PictureInPictureParams.Builder()
            .setAspectRatio(Rational(9, 16))

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            builder.setAutoEnterEnabled(autoEnterEnabled)
            builder.setSeamlessResizeEnabled(false)
        }

        return builder.build()
    }

    private fun applyPictureInPictureUi(isInPip: Boolean) {
        val overlayVisibility = if (isInPip) View.GONE else View.VISIBLE
        statusCard.visibility = overlayVisibility
        controlsCard.visibility = overlayVisibility
        if (currentOnboardingStep != OnboardingStep.DONE) {
            onboardingOverlay.visibility = overlayVisibility
        }
    }

    private fun setupActions() {
        btnOnboardingPrimary.setOnClickListener {
            when (currentOnboardingStep) {
                OnboardingStep.DEVICE_PLACEMENT -> showOnboardingStep(OnboardingStep.CALIBRATION)
                OnboardingStep.CALIBRATION -> completeOnboardingAndStartMonitoring()
                OnboardingStep.DONE -> Unit
            }
        }

        btnMonitoringToggle.setOnClickListener {
            if (isMonitoringActive || isCameraBound) {
                stopMonitoring(updateUserIntent = true)
            } else {
                startMonitoring(updateUserIntent = true)
            }
        }

        btnRecalibrate.setOnClickListener {
            recalibrate()
        }
    }

    private fun completeOnboardingAndStartMonitoring() {
        currentOnboardingStep = OnboardingStep.DONE
        onboardingOverlay.visibility = View.GONE
        statusCard.visibility = View.VISIBLE
        controlsCard.visibility = View.VISIBLE
        startMonitoring(updateUserIntent = false)
    }

    private fun showOnboardingStep(step: OnboardingStep) {
        currentOnboardingStep = step
        onboardingOverlay.visibility = View.VISIBLE
        statusCard.visibility = View.GONE
        controlsCard.visibility = View.GONE

        when (step) {
            OnboardingStep.DEVICE_PLACEMENT -> {
                ivOnboardingIllustration.setImageResource(R.drawable.ic_onboarding_camera_face)
                tvOnboardingStep.text = getString(R.string.onboarding_step_1)
                tvOnboardingTitle.text = getString(R.string.onboarding_place_title)
                tvOnboardingBody.text = getString(R.string.onboarding_place_body)
                btnOnboardingPrimary.text = getString(R.string.onboarding_ready)
            }
            OnboardingStep.CALIBRATION -> {
                ivOnboardingIllustration.setImageResource(R.drawable.ic_onboarding_calibration)
                tvOnboardingStep.text = getString(R.string.onboarding_step_2)
                tvOnboardingTitle.text = getString(R.string.onboarding_calibration_title)
                tvOnboardingBody.text = getString(R.string.onboarding_calibration_body)
                btnOnboardingPrimary.text = getString(R.string.onboarding_start_monitoring)
            }
            OnboardingStep.DONE -> {
                onboardingOverlay.visibility = View.GONE
                statusCard.visibility = View.VISIBLE
                controlsCard.visibility = View.VISIBLE
            }
        }
    }

    private fun initEngine() {
        try {
            drowsinessEngine?.close()
            drowsinessEngine = null
            drowsinessEngine = DrowsinessEngine.create(this)
            Log.d("MainActivity", "Drowsiness engine initialized successfully")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error loading model or stats: ${e.message}", e)
            tvStatus.text = "Lỗi nạp Model!"
        }
    }

    private fun setupMediaPipe() {
        val baseOptions = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
        val options = FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumFaces(1)
            .setResultListener { result, _ ->
                if (!isMonitoringActive) return@setResultListener
                if (result.timestampMs() < minAcceptedFrameTimestampMs) return@setResultListener

                drowsinessEngine?.let { engine ->
                    val snapshot = engine.processResult(result)
                    runOnUiThread {
                        if (!isMonitoringActive) return@runOnUiThread
                        if (result.timestampMs() < minAcceptedFrameTimestampMs) return@runOnUiThread

                        renderSnapshot(snapshot)
                        if (snapshot.state == DetectionState.ALERT) {
                            alertManager.triggerAlarm()
                        }
                    }
                }
            }
            .setErrorListener { error ->
                Log.e("MediaPipe", "Error: ${error.message}")
            }
            .build()
        
        faceLandmarker = FaceLandmarker.createFromOptions(this, options)
    }

    private fun startCamera() {
        if (isCameraBinding || isCameraBound) return

        isCameraBinding = true
        isMonitoringActive = true
        val sessionToken = ++monitoringSessionToken
        minAcceptedFrameTimestampMs = TimeUnit.NANOSECONDS.toMillis(SystemClock.elapsedRealtimeNanos())
        lastProcessedFrameTimestampMs = 0L
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            if (!isMonitoringActive || sessionToken != monitoringSessionToken) {
                isCameraBinding = false
                return@addListener
            }

            val cameraProvider = try {
                cameraProviderFuture.get()
            } catch (e: Exception) {
                isCameraBinding = false
                isCameraBound = false
                isMonitoringActive = false
                Log.e("CameraX", "Unable to get camera provider", e)
                showCameraStartErrorUi()
                return@addListener
            }
            this.cameraProvider = cameraProvider

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(viewFinder.surfaceProvider)
            }

            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        if (!isMonitoringActive || sessionToken != monitoringSessionToken) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        val bitmap = imageProxy.toBitmap()
                        val mpImage = BitmapImageBuilder(bitmap).build()
                        val frameTimestampMs = TimeUnit.NANOSECONDS.toMillis(imageProxy.imageInfo.timestamp)
                        if (frameTimestampMs < minAcceptedFrameTimestampMs) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        if (frameTimestampMs - lastProcessedFrameTimestampMs < FRAME_PROCESSING_INTERVAL_MS) {
                            imageProxy.close()
                            return@setAnalyzer
                        }

                        lastProcessedFrameTimestampMs = frameTimestampMs
                        faceLandmarker?.detectAsync(mpImage, frameTimestampMs)
                        imageProxy.close()
                    }
                }

            val cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalyzer)
                isCameraBound = true
            } catch (exc: Exception) {
                isCameraBound = false
                Log.e("CameraX", "Use case binding failed", exc)
            } finally {
                isCameraBinding = false
            }

        }, ContextCompat.getMainExecutor(this))
    }

    private fun startMonitoring(updateUserIntent: Boolean = true) {
        if (updateUserIntent) {
            isMonitoringRequestedByUser = true
        }

        if (!hasCameraPermission()) {
            showCameraPermissionRequiredUi()
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            return
        }

        btnMonitoringToggle.text = getString(R.string.stop_monitoring)
        startCamera()
    }

    private fun recalibrate() {
        if (currentOnboardingStep != OnboardingStep.DONE) return

        isMonitoringRequestedByUser = true
        stopCamera()
        initEngine()
        tvAlertBanner.text = getString(R.string.banner_calibrating)
        tvAlertBanner.setBackgroundColor(Color.parseColor("#CC2563EB"))
        tvStatus.text = getString(R.string.waiting_for_calibration)
        tvGuidance.text = getString(R.string.guidance_calibrating)
        progressStatus.visibility = View.VISIBLE
        progressStatus.progress = 0
        startMonitoring(updateUserIntent = false)
    }

    private fun stopMonitoring(updateUserIntent: Boolean = true) {
        if (updateUserIntent) {
            isMonitoringRequestedByUser = false
        }

        stopCamera()
        showMonitoringPausedUi()
    }

    private fun stopCamera() {
        monitoringSessionToken++
        isMonitoringActive = false
        cameraProvider?.unbindAll()
        isCameraBound = false
        isCameraBinding = false
        alertManager.stopAll()
    }

    private fun hasCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
    }

    private fun showCameraPermissionRequiredUi() {
        progressStatus.visibility = View.GONE
        tvAlertBanner.text = getString(R.string.banner_camera_required)
        tvAlertBanner.setBackgroundColor(Color.parseColor("#CC334155"))
        tvStatus.text = getString(R.string.camera_permission_required)
        tvGuidance.text = getString(R.string.guidance_camera_permission)
    }

    private fun showCameraStartErrorUi() {
        progressStatus.visibility = View.GONE
        tvAlertBanner.text = getString(R.string.banner_camera_error)
        tvAlertBanner.setBackgroundColor(Color.parseColor("#CCB91C1C"))
        tvStatus.text = getString(R.string.camera_start_error)
        tvGuidance.text = getString(R.string.guidance_camera_error)
    }

    private fun showMonitoringPausedUi() {
        progressStatus.visibility = View.GONE
        btnMonitoringToggle.text = getString(R.string.start_monitoring)
        tvAlertBanner.text = getString(R.string.banner_paused)
        tvAlertBanner.setBackgroundColor(Color.parseColor("#CC1E293B"))
        tvStatus.text = getString(R.string.status_monitoring_paused)
        tvGuidance.text = getString(R.string.guidance_resume_monitoring)
    }

    private fun renderSnapshot(snapshot: DetectionSnapshot) {
        tvAlertBanner.text = when (snapshot.state) {
            DetectionState.CALIBRATING -> getString(R.string.banner_calibrating)
            DetectionState.BUFFERING -> getString(R.string.banner_buffering)
            DetectionState.MONITORING -> getString(R.string.banner_monitoring)
            DetectionState.WARNING -> getString(R.string.banner_warning)
            DetectionState.ALERT -> getString(R.string.banner_alert)
            DetectionState.FACE_LOST -> getString(R.string.banner_face_lost)
        }
        tvAlertBanner.setBackgroundColor(
            when (snapshot.state) {
                DetectionState.CALIBRATING -> Color.parseColor("#CC2563EB")
                DetectionState.BUFFERING -> Color.parseColor("#CC0891B2")
                DetectionState.MONITORING -> Color.parseColor("#CC15803D")
                DetectionState.WARNING -> Color.parseColor("#CCEA580C")
                DetectionState.ALERT -> Color.parseColor("#CCB91C1C")
                DetectionState.FACE_LOST -> Color.parseColor("#CC334155")
            }
        )
        tvStatus.text = snapshot.statusText
        tvGuidance.text = when (snapshot.state) {
            DetectionState.CALIBRATING -> getString(R.string.guidance_calibrating)
            DetectionState.BUFFERING -> getString(R.string.guidance_buffering)
            DetectionState.MONITORING -> getString(R.string.guidance_monitoring)
            DetectionState.WARNING -> getString(R.string.guidance_warning)
            DetectionState.ALERT -> getString(R.string.guidance_alert)
            DetectionState.FACE_LOST -> getString(R.string.guidance_face_lost)
        }
        renderStatusProgress(snapshot)
    }

    private fun renderStatusProgress(snapshot: DetectionSnapshot) {
        when (snapshot.state) {
            DetectionState.CALIBRATING -> {
                progressStatus.visibility = View.VISIBLE
                progressStatus.progress = calibrationProgress(snapshot.statusText)
            }
            DetectionState.BUFFERING -> {
                val progress = bufferingProgress(snapshot.statusText)
                progressStatus.visibility = if (progress != null) View.VISIBLE else View.GONE
                if (progress != null) progressStatus.progress = progress
            }
            else -> progressStatus.visibility = View.GONE
        }
    }

    private fun calibrationProgress(statusText: String): Int {
        val timeLeftSeconds = Regex("\\((\\d+(?:[,.]\\d+)?) s\\)")
            .find(statusText)
            ?.groupValues
            ?.getOrNull(1)
            ?.replace(',', '.')
            ?.toFloatOrNull()
            ?: return 0

        val elapsedRatio = 1f - (timeLeftSeconds / 5f).coerceIn(0f, 1f)
        return (elapsedRatio * 100).toInt().coerceIn(0, 100)
    }

    private fun bufferingProgress(statusText: String): Int? {
        val match = Regex("\\((\\d+)/(\\d+)\\)").find(statusText) ?: return null
        val collected = match.groupValues.getOrNull(1)?.toIntOrNull() ?: return null
        val required = match.groupValues.getOrNull(2)?.toIntOrNull() ?: return null
        if (required <= 0) return null

        return ((collected.toFloat() / required.toFloat()) * 100).toInt().coerceIn(0, 100)
    }

    override fun onDestroy() {
        super.onDestroy()
        stopCamera()
        cameraExecutor.shutdown()
        faceLandmarker?.close()
        drowsinessEngine?.close()
        alertManager.release()
    }
}
