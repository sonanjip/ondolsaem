package io.github.taehunkim.ondolsaem.sdk

/** SenseVoice language hints (API 5). These are input-language constraints, not translation targets. */
object SpeechLanguages {
    const val AUTO = "auto"
    val supported: Set<String> = setOf(AUTO, "ko", "en", "ja", "zh", "yue")
    fun requireSupported(language: String) {
        require(language in supported) { "Unsupported speech language: $language" }
    }
}
