# OndolSaem / 온돌샘

**English** | [한국어](README.ko.md)

OndolSaem is an Android app that hosts on-device AI models for other apps. It downloads curated models per category (text/image/audio embedding, chat LLM, OCR, speech recognition, faces, yes/no decisions, image generation), keeps them in memory according to a per-category policy, and serves them to client apps over a bound AIDL service. A model is loaded once for all apps instead of once per app.

The name joins *ondol* (온돌, the floor heating that warms a whole house from underneath) and *saem* (샘, a spring everyone draws water from). As acronyms:
- **ONDOL**: On-device Neural Deployment & Orchestration Layer
- **SAEM**: Shared AI Engine Manager

<p>
<img src="docs/images/demo.gif" width="260" alt="OndolSaem loads a model on request, lists the downloaded models and shows which apps use them">
</p>

*One model in memory for every app: load on demand, the model catalog, and the connected apps.*

Status: early development (0.9.0). The first client is [Nunsori / 눈소리](https://github.com/sonanjip/nunsori), an on-device semantic search app.

## Features

| Feature | Description |
|---|---|
| Speech recognition | SenseVoice with per-request automatic/Korean/English/Japanese/Mandarin/Cantonese hints (API 5, SDK 0.7.0). Existing clients retain automatic detection. |
| Model categories | Each category has a short description and curated models; one downloaded model is the category default |
| Chat LLM | Supports LiteRT-LM models (Gemma 4 E2B/E4B, Qwen3.5 2B VL, LFM3 VL/Omni 2B) and **Gemini Nano via Android AICore** on supported devices (text & images, ~1.1–1.5 GB system RAM, minimal host memory overhead) |
| Downloads | Warning before downloading on a metered network (mobile data or metered Wi-Fi), with a "wait for Wi-Fi" option |
| Model changes | Changing an embedding model warns which apps must re-embed their data; client apps are notified right away or on their next connection |
| Residency | Per category: keep loaded, unload right after use, or unload after N minutes. Unloaded models are loaded again on the next request. If Android refuses to keep the host running in the background, a notification and a dashboard card offer to turn off battery optimization |
| First run | A short guide: what OndolSaem is, connected apps, recommended models to download (740M embedding with audio by default), notifications |
| Dashboard | Device memory, the host's memory use, loaded models and which apps hold them |
| Apps | Apps that use the host, their categories and request statistics; own apps (same signing key or an allowed key) connect directly; other apps ask first and are allowed, denied or revoked here |
| Decisions | Category *Decisions*: a multilingual (Korean too) natural language inference model answers whether a text or image matches a topic or condition, with a probability. Apps use it to tag notes and to check new files for scheduled tasks. Models include mDeBERTa-v3 (325 MB int8 converted on-device, or 4-bit) for text, and LiquidAI d1-Omni 2B (~1.95 GB after on-device int8 conversion) for direct multimodal entailment (evaluates images directly without requiring OCR) |
| Image generation | Category *Image generation*: on-device text-to-image synthesis using Iris 3B Turbo (768×768 px, ~7.2 GB after on-device int8 conversion of the text encoder and embedding table). Unloads immediately after use to minimize memory footprint. Served over AIDL to client apps (e.g. Malbotdari) |
| NPU | OndolSaem checks the phone's chip and looks up NPU builds of its models for it on Hugging Face (today: Pixel 10/11, Snapdragon 8 Gen 2 to 8 Elite Gen 5, Dimensity 9400/9500); found ones can be downloaded under the model and it runs on the NPU, falling back to the GPU, then the CPU. Snapdragon phones also get Qualcomm's NPU runtime (about 25 MB). Settings › NPU › "Check the NPU" compares CPU, GPU and NPU on the phone and writes a report to share |
| Client SDK | `com.github.sonanjip:ondolsaem:<tag>` on JitPack, see [docs/architecture.md](docs/architecture.md) |

## Documentation

- [docs/architecture.md](docs/architecture.md): service contract, categories, residency, model-change notifications, access control
- [docs/development.md](docs/development.md): build, scripts, SDK publishing, working with a client app
- [docs/client-guide.md](docs/client-guide.md): using OndolSaem from an app
- [docs/privacy.md](docs/privacy.md): what is stored, network, who can connect
- [docs/roadmap.md](docs/roadmap.md): planned work

## About the name

**온돌샘 (OndolSaem)** joins **온돌** (*ondol*), the Korean floor heating that warms a whole house from underneath, and **샘** (*saem*), a spring that everyone draws water from. OndolSaem works the same way: it sits below the other apps, unseen, keeps the models warm in memory, and every app draws from the same source instead of carrying its own copy. *Ondol* and *saem* were two separate candidates; they were joined into one word so the name is distinctive, and together they make one picture: a spring rising from the warm stones under the house.

The two parts have their own backronyms: **ONDOL**, On-device Neural Deployment & Orchestration Layer (running and managing the models), and **SAEM**, Shared AI Engine Manager (sharing them). Say it "on-dol-sam".

## License

Apache License 2.0, see [LICENSE](LICENSE). Third-party components: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
