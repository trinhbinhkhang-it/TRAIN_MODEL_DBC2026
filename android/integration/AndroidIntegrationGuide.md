# Android Integration Guide - SwapFace Detector

## 1. Android Studio Setup
- Android Studio Arctic Fox+ | SDK API 26+ | Target API 34 | NDK 25+

## 2. Gradle Dependencies (app/build.gradle.kts)
```kotlin
dependencies {
    implementation("org.tensorflow:tensorflow-lite:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.16.1")
    implementation("org.tensorflow:tensorflow-lite-support:0.4.4")
    implementation("androidx.camera:camera-core:1.3.2")
    implementation("androidx.camera:camera-camera2:1.3.2")
    implementation("androidx.camera:camera-view:1.3.2")
    implementation("com.google.mediapipe:face_detection:0.10.14")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")
}
```

## 3. AndroidManifest Permissions
```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION" />
<uses-permission android:name="android.permission.FOREGROUND_SERVICE_CAMERA" />
<uses-permission android:name="android.permission.POST_NOTIFICATIONS" />
<uses-permission android:name="android.permission.SYSTEM_ALERT_WINDOW" />
```

## 4. Foreground Service (ProtectionService.kt)
Key components:
- MediaProjection + VirtualDisplay + ImageReader for screen capture
- MediaPipe Face Detection
- TFLite Interpreter (FP16 + GPU delegate)
- Temporal Filter (EMA + Hysteresis)
- Overlay Window (TYPE_APPLICATION_OVERLAY)

See full implementation in the guide.

## 5. Model Placement
```bash
cp model_files/swapface_detector_fp16.tflite app/src/main/assets/
```

## 6. Critical Limitations
**MediaProjection CANNOT capture:**
- FLAG_SECURE apps (banking, DRM video)
- System dialogs
- Your app MUST handle INPUT_UNAVAILABLE state

## 7. Build & Run
```bash
./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

Full implementation details in the main guide artifact.