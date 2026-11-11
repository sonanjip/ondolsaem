# Third-party notices

OndolSaem is licensed under the [Apache License 2.0](LICENSE), Copyright 2026 Taehun Kim.
It uses the components below. Versions come from [`gradle/libs.versions.toml`](gradle/libs.versions.toml).

## Libraries (linked into the app)

| Component | Version | License |
|---|---|---|
| AndroidX (Compose BoM, Material 3, Activity, Lifecycle, WorkManager, DataStore, AppCompat) | see catalog (Compose BoM 2026.09.00, WorkManager 2.12.0) | Apache-2.0 |
| Kotlin, kotlinx.coroutines | 2.4.21 / 1.11.0 | Apache-2.0 |
| OkHttp | 5.5.1 | Apache-2.0 |
| LiteRT-LM (`litertlm-android`) | 0.18.1 | Apache-2.0 |
| LiteRT (`litert`, `litert-gpu`) | 2.3.2 | Apache-2.0 |
| LiteRT NPU runtimes (`litert-npu-runtime-google-tensor`, `-qualcomm`, `-mediatek`; dispatch libraries only) | 2.3.2 | Apache-2.0 |
| sherpa-onnx (AAR, downloaded by `scripts/fetch-sherpa-onnx.sh`, not committed) | 2.0.1 | Apache-2.0 |
| Apache Commons Compress | 1.28.0 | Apache-2.0 |
| ML Kit Face Detection (bundled model) | 16.1.7 | [ML Kit Terms of Service](https://developers.google.com/ml-kit/terms) (proprietary; fetched through Gradle, not redistributed in this repository) |
| ML Kit GenAI Prompt API (`genai-prompt`) | 1.0.0-beta5 | [ML Kit Terms of Service](https://developers.google.com/ml-kit/terms) (client SDK talking to Android AICore) |
| ONNX Runtime C API headers (`app/src/main/cpp/include/onnxruntime`, v1.28.0; the runtime itself comes with sherpa-onnx) | 1.28.0 | MIT |
| ML Kit Text Recognition (Korean) | 16.0.1 | [ML Kit Terms of Service](https://developers.google.com/ml-kit/terms) (proprietary; fetched through Gradle, not redistributed in this repository) |

Downloaded at run time, not in the APK or this repository: Qualcomm QNN runtime (`com.qualcomm.qti:qnn-runtime` 2.51.0 from Maven Central), only on Snapdragon phones with an NPU build, under the [Qualcomm AI Hub Model License](https://softwarecenter.qualcomm.com/api/download/software/licenses/ai_model_hub/v1/LICENSE.pdf) stated in its POM.

The client SDK (`:sdk`) uses only kotlinx.coroutines and the Android framework.

## Models

No model is stored in this repository. The app downloads them at the user's request (Models tab), except Silero VAD,
which is fetched at build time and bundled in the APK. Each model keeps its own license; check it before
redistributing.

| Model | Size | Source | License as stated by the source |
|---|---|---|---|
| EmbeddingGemma 2 740M (+audio) | ~485 MB | Hugging Face `litert-community/embeddinggemma-2-740m-litert-lm` | Apache-2.0 (model card) |
| NPU builds of EmbeddingGemma 2 740M and Gemma 4 E2B (one chip each) | 0.9 GB / 3.2–3.5 GB | the same repositories (`*_Qualcomm_SM8750.litertlm`, `*_Google_Tensor_G5.litertlm`, …) | Apache-2.0 (model card) |
| Gemma 4 E2B / E4B (chat) | 2.6 GB / 3.7 GB | Hugging Face `litert-community/gemma-4-E2B-it-litert-lm`, `gemma-4-E4B-it-litert-lm` | Apache-2.0 (model card) |
| Qwen3.5 2B VL (chat, images) | 2.9 GB | Hugging Face `litert-community/Qwen3.5-2B` | Apache-2.0 |
| LFM3 VL 2B (chat, images) | 1.9 GB | Hugging Face `litert-community/LFM3-VL-2B-LiteRT` | LFM Open License 1.0 ("Built with LFM") |
| LFM3 Omni 2B (chat, images, audio) | 3.0 GB | Hugging Face `litert-community/LFM3-Omni-2B-LiteRT` | LFM Open License 1.0 ("Built with LFM") |
| Gemini Nano (chat, images via Android AICore) | n/a (system-managed) | Android AICore (`com.google.android.aicore`) | ML Kit Terms |
| SenseVoice small, int8 (via sherpa-onnx) | 163 MB archive | GitHub `k2-fsa/sherpa-onnx` release `asr-models` | FunASR Model License |
| Silero VAD | 0.6 MB | GitHub `k2-fsa/sherpa-onnx` release `asr-models` | MIT |
| ML Kit Korean text recognition | n/a | Google (Gradle dependency) | ML Kit Terms |
| SFace (face recognition, `face_recognition_sface_2021dec.onnx`) | 39 MB | Hugging Face `opencv/face_recognition_sface` (OpenCV Zoo) | Apache-2.0 (LICENSE in the repository) |
| ML Kit face detection | n/a | Google (Gradle dependency) | ML Kit Terms |
| mDeBERTa-v3-base xnli multilingual NLI (decision; fp32 converted on the phone to int8, or q4) | 1.1 GB download → 325 MB / 845 MB | Hugging Face `onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX` (from `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`) | MIT (model card) |
| LiquidAI d1-Omni 2B (decision, text & images) | 3.6 GB download | Hugging Face `litert-community/d1-Omni-2B-LiteRT` | LFM Open License 1.0 ("Built with LFM") |
| Iris 3B Turbo (image generation, 768px) | 10.5 GB | Hugging Face `litert-community/iris-3b-Turbo-LiteRT` | Apache-2.0 |

The EmbeddingGemma model cards say Apache-2.0 but also link to the Gemma terms of use; the catalog in
[`ModelCatalog.kt`](app/src/main/java/io/github/sonanjip/ondolsaem/catalog/ModelCatalog.kt) follows the card.
