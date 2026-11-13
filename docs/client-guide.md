# Using OndolSaem from an app

1. Depend on the SDK: `implementation("com.github.sonanjip:ondolsaem:sdk-v0.5.0")` (the version is the git tag) with `maven("https://jitpack.io")` in the repositories (or a composite build of a local checkout, see [development.md](development.md#working-with-a-client-app)). The SDK's manifest adds the `<queries>` entry for the host and the model-change receiver.
   - Apps not signed with one of the host's keys must be approved by the user first. Their first call fails with `NOT_APPROVED` and OndolSaem shows a request; send the user there with `startActivity(OndolSaem.approvalIntent())` and retry after they return. A denied app gets the same reason. Clients built with SDK 0.1.0 see `NOT_ALLOWED` instead.
2. Sign the app with the same key as OndolSaem, or add its certificate digest to the host's `allowed_signers.xml`.
3. Create one `OndolSaemClient(context)` per process. It binds on first use and unbinds when idle.

```kotlin
val client = OndolSaemClient(context)
if (!client.isInstalled()) { /* show an install hint */ }
val info = client.activeModel(OndolSaem.Categories.EMBEDDING)   // id, version, ready, audio, vision
val v = client.embedQuery("a cat on a sofa")                     // EmbeddingResult(vector, modelId, modelVersion)
client.acquire(OndolSaem.Categories.EMBEDDING).use { /* model stays loaded for a batch */ }
client.chat("conv-1", system, history, "question", tools = listOf(ToolSpec(...))) { name, args -> runTool(name, args) }
    .collect { event -> /* Token / ToolRan */ }
if (client.supports(OndolSaem.Api.V2)) {
    val faces = client.detectFaces(jpeg)                          // FaceResult(left, top, right, bottom, score, vector, modelId, modelVersion)
}
if (client.supports(OndolSaem.Api.V3)) {
    val p = client.entail(noteText, listOf(Decisions.hypothesis("여행"), Decisions.hypothesis("요리", "레시피, 음식")))
    val travel = Decisions.verdict(p[0])                          // YES (>= 0.70), NO (<= 0.30) or UNSURE
}
```

- Decisions (API 3): one call per text with all its questions (up to 64). Store the decision model id (`activeModel(DECISION).id`) next to derived answers such as tags; a new model may answer differently. Text only: images need their OCR text or another route.

- Faces (API 2): compare `vector`s with the cosine similarity (0.363 or more: likely the same person). Face vectors are biometric data; keep them on the device and let the user delete them.

- Store `modelId` (and `modelVersion`) next to anything derived from a model. When it differs from `activeModel()` the data is stale.
- React to model changes: collect `client.modelChanges` while running, and set `ModelChangedReceiver.handler` in `Application.onCreate` for changes while the app is not running (treat it as a hint and re-check `activeModel()`).
- Errors: `HostUnavailableException` (`NOT_INSTALLED`, `NOT_ALLOWED`, `NOT_APPROVED`, `NO_ANSWER`, `HOST_TOO_OLD`, `CLIENT_TOO_OLD`) and `HostCallException` with `OndolSaem.Errors` codes (`MODEL_NOT_READY` means: ask the user to download or pick a model in OndolSaem).
- Use `PRIORITY_BACKGROUND` for bulk work so interactive requests from any app go first; background requests of several apps take turns, one request each.
- Versions: the client exchanges versions with the host on every connection (`hostInfo()` returns the host's level, minimum client level and app version). Pass `minHostApi` to `OndolSaemClient` for the lowest host level your app works with; calls into newer levels are checked by the SDK. Handle `HOST_TOO_OLD` ("update OndolSaem") and `CLIENT_TOO_OLD` ("update this app"); use `supports(OndolSaem.Api.Vx)` to switch optional features off on older hosts.
- Background work "when idle": `IdleRule` / `IdlePreset` (SDK 0.5.0) define idle the same way for every app: charging and the screen off for 10, 30 or 60 minutes at ≥ 90 %, 70–89 % or < 70 % battery (preset `NORMAL`; `FAST` 5/15/30, `RELAXED` 30/60/120). `IdleState(context)` reads the battery, the charger and the screen (`isIdle(preset)`, `screenOn()`); call `IdleState(this).watchScreen()` in `Application.onCreate` for exact screen-off times. Pattern used by Nunsori and Jogakbo: a 15-minute periodic WorkManager job with only the charging constraint, which returns at once unless `isIdle`, and stops its loop when `screenOn()`.

## Speech language (API 5, SDK 0.7.0 / host 0.9.0)

```kotlin
// Legacy auto remains available on API 1+.
client.transcribe(pcm, 0L) { startMs, endMs, text -> /* consume cue */ }
// Request-scoped input language, not a translation target (API 5).
if (client.supports(OndolSaem.Api.V5)) {
    client.transcribeWithLanguage(pcm, 0L, "ko") { startMs, endMs, text -> /* consume cue */ }
}
```

Supported codes: `auto`, `ko`, `en`, `ja`, `zh` (Mandarin), `yue` (Cantonese). SYSTEM is a client-side preference: resolve it before calling. Unsupported SDK values throw IllegalArgumentException; invalid raw Binder requests return BAD_INPUT. Explicit-language calls to older hosts report HOST_TOO_OLD; the SDK never silently discards a hint. Legacy calls always use auto, including after another client requested a fixed language. PCM format, cue streaming, resume offsets, priority and cancellation remain unchanged.
