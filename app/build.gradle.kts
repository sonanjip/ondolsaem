plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

// Single place to bump the app version: major.minor.patch, versionCode = major*10000 + minor*100 + patch.
val appVersionName = "0.6.0"
val appVersionCode = appVersionName.split(".").let { (a, b, c) -> a.toInt() * 10000 + b.toInt() * 100 + c.toInt() }

android {
    namespace = "io.github.taehunkim.ondolsaem"
    compileSdk = 37

    defaultConfig {
        applicationId = "io.github.taehunkim.ondolsaem"
        minSdk = 29
        targetSdk = 36
        versionCode = appVersionCode
        versionName = appVersionName
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        ndk { abiFilters += listOf("arm64-v8a") }
    }
    // Release signing: the key lives outside the repo (see docs/development.md, "Release build").
    // Without these properties `assembleRelease` still builds, but the APK is unsigned.
    fun signingProp(name: String): String? = (project.findProperty(name) as String?) ?: System.getenv(name)
    val releaseStore = signingProp("ONDOLSAEM_STORE_FILE")
    if (releaseStore != null) {
        signingConfigs.create("release") {
            storeFile = file(releaseStore)
            storePassword = signingProp("ONDOLSAEM_STORE_PASSWORD")
            keyAlias = signingProp("ONDOLSAEM_KEY_ALIAS")
            keyPassword = signingProp("ONDOLSAEM_KEY_PASSWORD")
        }
        buildTypes.getByName("release").signingConfig = signingConfigs.getByName("release")
    }
    buildFeatures { compose = true; buildConfig = true }
    androidResources { generateLocaleConfig = true }
    packaging {
        jniLibs {
            // extracted to nativeLibraryDir: the NPU backend loads its dispatch library from there (engine/Npu.kt)
            useLegacyPackaging = true
            // on-device NPU compilers: only ahead-of-time compiled NPU builds are used
            excludes += "**/libLiteRtCompilerPlugin_*.so"
        }
    }
    // JNI bridge to sherpa-onnx's ONNX Runtime for models sherpa does not cover (SFace), see src/main/cpp.
    externalNativeBuild { cmake { path = file("src/main/cpp/CMakeLists.txt"); version = "3.31.6" } }
    ndkVersion = "28.2.13676358"
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    implementation(project(":sdk"))
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.lifecycle.runtime.compose)
    implementation(libs.androidx.appcompat)
    implementation(libs.androidx.compose.material.icons)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.work.runtime)
    implementation(libs.androidx.datastore.preferences)
    implementation(libs.okhttp)
    implementation(libs.commons.compress)
    implementation(libs.litertlm.android)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    // LiteRT NPU dispatch libraries (Apache-2.0, ~0.5 MB each); Qualcomm's QNN runtime is downloaded on demand (NpuRuntime)
    implementation(libs.litert.npu.google.tensor)
    implementation(libs.litert.npu.qualcomm)
    implementation(libs.litert.npu.mediatek)
    implementation(libs.mlkit.text.recognition.korean)
    implementation(libs.mlkit.face.detection)
    // Not on Maven Central: scripts/fetch-sherpa-onnx.sh downloads it into app/libs (git-ignored).
    implementation(files("libs/sherpa-onnx-2.0.1.aar"))
    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.kotlinx.coroutines.play.services)

    testImplementation(libs.junit)
    testImplementation(libs.org.json) // the real org.json (android.jar has stubs)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(libs.okhttp.mockwebserver)
    androidTestImplementation(libs.androidx.test.ext.junit)
    androidTestImplementation(libs.androidx.test.runner)
    androidTestImplementation(libs.kotlinx.coroutines.test)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
}

// Decision model checks against desktop references (skipped without them):
// -PnliTokenizer=tokenizer.json (UnigramTokenizerTest), -PnliFp32=model.onnx -PnliW8Check=weights.bin made by
// scripts/make-nli-w8-template.py --check (NliW8ConverterTest)
tasks.withType<Test>().configureEach {
    providers.gradleProperty("nliTokenizer").orNull?.let { environment("NLI_TOKENIZER", it) }
    providers.gradleProperty("nliFp32").orNull?.let { environment("NLI_FP32", it) }
    providers.gradleProperty("nliW8Check").orNull?.let { environment("NLI_W8_CHECK", it) }
}
