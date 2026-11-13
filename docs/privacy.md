# Privacy

OndolSaem has no server, no account and no analytics. It runs AI models on the phone for other apps.

## What is stored

In the app's private storage (removed on uninstall or "Clear data"):

- Downloaded models under `files/models/<model id>/` (NPU builds in its `npu/` folder), the last NPU build search (`files/npu-builds.json`: chip, time, found files), and the NPU libraries under `files/npu/` (a copy of the vendor's dispatch library; on Snapdragon phones also Qualcomm's QNN runtime, about 25 MB).
- Settings (DataStore `host_settings`): the active model and memory policy of each category, the chat context size, whether to use the NPU, the optional Hugging Face token, the list of client apps (package names and the categories each used), and the decisions about other apps (package, signing-certificate SHA-256, allowed/denied/waiting, time of the last request). Request counts and timings are kept in memory only.

Inputs that client apps send (texts, images, audio, chat messages) are processed in memory and never stored (this includes photos sent for face detection: OndolSaem returns the face boxes and vectors to the app and keeps nothing); speech input is copied to a temporary file in the cache for the duration of the call and deleted afterwards. Chat conversation state stays in memory while the chat model is loaded. When **Gemini Nano (`chat-gemini-nano`)** is selected, chat prompts and attached images are sent locally over Android IPC to the system `com.google.android.aicore` process for on-device inference (still never leaving the phone).

## Network

Only to download models the user asks for:

- `https://huggingface.co/litert-community/...` (EmbeddingGemma 2 and the chat models; the Hugging Face token, if set, is sent only here)
- `https://github.com/k2-fsa/sherpa-onnx/releases/...` (SenseVoice)
- `https://huggingface.co/opencv/face_recognition_sface/...` (SFace)
- `https://huggingface.co/onnx-community/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7-ONNX/...` (decision model and its tokenizer)
- `https://huggingface.co/api/models/<repo>/tree/main` for the repositories of the LiteRT-LM models, when the app is opened with NPU use on (at most weekly) or on "Check again": lists their files to find NPU builds for this phone's chip. The chip name is not sent; the Hugging Face token, if set, is.
- `https://repo1.maven.org/maven2/com/qualcomm/qti/qnn-runtime/...` (Qualcomm's NPU runtime, only on Snapdragon phones with the first NPU build; no token or other data is sent)
- **Android AICore** (`com.google.android.aicore`): when the user installs Gemini Nano (`chat-gemini-nano`), Android AICore downloads the system model in the background through Google Play / AICore rather than OndolSaem's HTTP downloader.

Before a download on a metered network (mobile data or metered Wi-Fi) the app asks first. Every downloaded file is checked against a SHA-256 recorded in the app, so a damaged or replaced file is not used.

## Who can use it

Apps signed with the same key as OndolSaem, or with a key listed in `app/src/main/res/values/allowed_signers.xml`, connect directly. Any other app must be allowed by the user: its first call is refused and OndolSaem shows a notification and an entry in the Apps tab; it can use the models only after the user taps Allow, and stops as soon as the user revokes it. A decision applies to one signing key: the same package signed with another key asks again. Every call is checked. Allowed apps send their inputs to OndolSaem, which processes them on the phone as described above.

## Permissions

| Permission | Why |
|---|---|
| `INTERNET`, `ACCESS_NETWORK_STATE` | Model downloads; metered-network check |
| `POST_NOTIFICATIONS` | Download progress and the "models kept in memory" notification |
| `FOREGROUND_SERVICE`, `FOREGROUND_SERVICE_DATA_SYNC` | Downloads in the background |
| `FOREGROUND_SERVICE_SPECIAL_USE` | Keeping models the user set to "keep loaded" in memory |
| `REQUEST_IGNORE_BATTERY_OPTIMIZATIONS` | Asking the user to turn off battery optimization, offered only after Android refused to keep models loaded from the background; it lets the keep-alive service start in the background |
| `com.google.android.apps.aicore.service.BIND_SERVICE` | Merged from `com.google.mlkit:genai-prompt` to bind Android AICore when Gemini Nano is used |

Speech requests may include a language code; it is used only for local SenseVoice inference. It is not saved as a host-wide setting or sent over the network.
