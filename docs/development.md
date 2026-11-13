# Development

Design: [architecture.md](architecture.md).

## Setup

- Android Studio (its bundled JBR is the JDK), SDK at `~/Library/Android/sdk` with API 37 platform, NDK 28.2 and CMake 3.31.6 (the app builds a small JNI bridge, see [architecture.md](architecture.md#faces)). `source scripts/env.sh` sets `JAVA_HOME`, `ANDROID_HOME` and `PATH`.
- `local.properties` holds `sdk.dir` (git-ignored).

## Layout

| Module | Contents |
|---|---|
| `:sdk` | Client contract (AIDL in `src/main/aidl`, constants in `OndolSaem`) and client library. Its manifest adds a `<queries>` entry for the host package to every client app. |
| `:app` | The host app (code map in [architecture.md](architecture.md)). Before the first build run `scripts/fetch-sherpa-onnx.sh` (sherpa-onnx AAR into `app/libs`, Silero VAD into `app/src/main/assets`; neither is committed). |

## Build and test

| Level | Command |
|---|---|
| Unit | `./gradlew assembleDebug testDebugUnitTest` |
| Instrumented | `./gradlew connectedDebugAndroidTest` (needs a device; uninstalls the app at the end, which deletes downloaded models). `HostServiceTest` drives the real bound service through `OndolSaemClient` with fake engines (`ModelRuntime`'s factory and `readyOverride` hooks), so it needs no model. `FaceEngineTest` runs the real face engine when SFace is downloaded and the photos of Nunsori's `scripts/fetch-face-fixtures.sh` are copied to the app's `files/faces` (`adb push` to `/data/local/tmp`, then `run-as ... cat >`); it logs the cosine of every pair (tag `OndolSaem.FaceTest`). `NliDeciderTest` runs the decision models that are downloaded on sentences with obvious answers and logs probabilities and times (tag `OndolSaem.NliTest`). |
| Decision model references | `UnigramTokenizerTest` and `NliW8ConverterTest` compare with desktop references when given them: `./gradlew testDebugUnitTest -PnliTokenizer=<tokenizer.json> -PnliFp32=<fp32 model.onnx> -PnliW8Check=<weights.bin>` (the last one from `NLI_CHECK_DIR=<dir> uv run --with onnx --with numpy python scripts/make-nli-w8-template.py <fp32 model.onnx> --check`). Fixtures: `uv run --with tokenizers python scripts/make-tokenizer-fixtures.py <tokenizer.json>`. |

Scripts (`scripts/`): `fetch-sherpa-onnx.sh`, `make-nli-w8-template.py` (rebuilds `assets/nli-w8` when the decision model changes), `make-tokenizer-fixtures.py`, `emu-start.sh [--window]`, `emu-stop.sh`, `run.sh` (installDebug + launch), `logcat.sh [N]`, `screenshot.sh`.

To let a client built on another machine (another debug key) connect, add that key's SHA-256 to `app/src/main/res/values/allowed_signers.xml` (`apksigner verify --print-certs app-debug.apk`). The emulator AVD `lsa_api36` is shared with Nunsori.

## Working with a client app

Clients depend on `com.github.sonanjip:ondolsaem:<tag>` (see below). To change the SDK and a client together, check this repository out next to the client and point the client at it. Nunsori, for example, reads `ondolsaem.dir=../ondolsaem` from its `local.properties`, adds this build as a Gradle composite build and substitutes `com.github.sonanjip:ondolsaem` with `project(":sdk")` (`dependencySubstitution` in its `settings.gradle.kts`), because the JitPack coordinates differ from the SDK's own `group`/name.

## Measuring on a device

`scripts/bench-device.sh [-s SERIAL] [-n REQUESTS]` runs `SharedModelBench` (androidTest, skipped unless the argument `bench=1` is given) on a connected phone or emulator with the active embedding model already downloaded. It installs the app and the test APK and starts the benchmark with `am instrument`, so the models stay (`connectedDebugAndroidTest` would uninstall the app). It writes `build/bench/<device>-<model>-<date>.txt`:

| Line | Meaning |
|---|---|
| `load_ms`, `backend`, `model_kb`, `pss_kb_before/after` | First load of the model, GPU or CPU, memory growth while loading, process PSS |
| `one client, query` / `image` | Latency of one client, one request at a time |
| `two clients, client A/B`, requests per second | Two clients sending background requests at once: each waits for the other's request, throughput stays the same |
| `interactive next to images` | Queries from one client while another sends images: a query waits at most for the one request already running (no preemption) |

The two clients run in the test process, so they share one app identity; the per-app turns of `PriorityGate` are covered by its unit tests. Emulator reference (CPU, eg2-440m, n=12): load 0.3 s, query p50 35 ms, image p50 0.73 s, two clients 28 requests/s, interactive next to images p50 0.76 s. Real-device GPU numbers still need to be recorded here (roadmap).

### NPU builds

Without a computer connection (a headless build machine, for example): install the debug APK on the phone, download an embedding model and its NPU build, then Settings › NPU › "Check the NPU" and share the report (see [architecture.md](architecture.md#npu)). `NpuSelfCheckTest` runs the check on the emulator, where the NPU must be reported as not running.

`VectorDumpTest` (`-e dumpTag <name>`) writes CPU vectors of fixed texts to the app cache, to compare two LiteRT-LM versions before switching (a change must not alter vectors, or clients would have to re-index).

With adb:

`scripts/npu-check.sh [-s SERIAL]` runs `NpuParityTest` on a connected phone the same way (the app stays installed). Download an embedding model and its NPU build in the app first (Models › the model › "Download NPU build"). For each embedding model with an NPU build it checks that the build loads on the NPU and that its vectors match the CPU build (cosine ≥ 0.98 for documents, queries and an image), and logs the cosine and the NPU and GPU time per input. It writes `build/npu/<device>-<date>.txt`. Without an NPU build (the emulator, other chips) the tests are skipped. `NpuTest` (unit) covers chip names, file-name matching and the search result's JSON; `NpuDiscoveryTest` (androidTest, needs the network) searches the real repositories for SM8750, Tensor G5 and SM8450. Unit tests use the real `org.json` (`testImplementation`), since `android.jar` only has stubs.

## Publishing the SDK

JitPack builds the SDK from a git tag (`jitpack.yml` runs `:sdk:publishReleasePublicationToMavenLocal`):

1. Bump `version` in `sdk/build.gradle.kts`.
2. Tag and push: `git tag -a sdk-vX.Y.Z -m "OndolSaem SDK X.Y.Z"`, `git push origin sdk-vX.Y.Z`. Clients use the tag name as the version.
3. Local check without JitPack: `./gradlew :sdk:publishReleasePublicationToMavenLocal` (writes to `~/.m2`).
4. Open `https://jitpack.io/com/github/sonanjip/ondolsaem/<tag>/build.log` to start the build and read its log (status: `https://jitpack.io/api/builds/com.github.sonanjip/ondolsaem/<tag>`).

JitPack finds the one artifact the build publishes (`com.github.taehunkim.ondolsaem:sdk`) and serves it as `com.github.sonanjip:ondolsaem:<tag>` (`.aar`, `.pom`, `.module`). Checked with `sdk-v0.2.0`: JitPack's `openjdk21` builds it (Gradle 9.8.2, AGP 9.5; a JDK 25 is not needed), and Nunsori builds from that artifact. Since `sdk-v0.3.0` the `:app` module declares an NDK (JNI bridge); JitPack installs NDK 28.2 on its own while configuring the build (about a minute more) and still builds only `:sdk`; Nunsori 0.9.0 builds from `sdk-v0.3.0`.

## Release build

Same scheme as Nunsori, with its own property names: `ONDOLSAEM_STORE_FILE`, `ONDOLSAEM_STORE_PASSWORD`, `ONDOLSAEM_KEY_ALIAS`, `ONDOLSAEM_KEY_PASSWORD` in `~/.gradle/gradle.properties` (or environment variables). Use the **same keystore and key as Nunsori**, so the service's signature check accepts release builds of both apps. Without these properties `./gradlew assembleRelease` builds an unsigned APK. Bump `appVersionName` in `app/build.gradle.kts` first.

ASR language checks: `HostServiceTest#explicitSpeechLanguageDoesNotChangeLegacyAutoAndCanStopEarly` uses fake engines. `SpeechLanguageSmokeTest` is opt-in (`-e realAsr true`), requires a downloaded SenseVoice model plus raw 16 kHz mono PCM files `files/asr-language-en.pcm` and `files/asr-language-ko.pcm` in the host target app; compares auto → English → Korean → legacy auto. Never run real models alongside builds on the Mac mini.

Verified 2026-11-14: HostServiceTest 25 tests + opt-in real SenseVoice language smoke 1 test passed. Auto recognition returned to its original result after fixed en/ko requests. Nunsori builds and its 170 JVM tests pass with the updated composite SDK. See sibling `../dorangori/docs/language-selection-test.md` for UI/upgrade checks.
