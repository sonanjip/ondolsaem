package io.github.taehunkim.ondolsaem.sdk

import org.junit.Assert.assertEquals
import org.junit.Test

class SpeechLanguagesTest {
    @Test fun allModelLanguagesAndAutoAreAccepted() {
        listOf("auto", "ko", "en", "ja", "zh", "yue").forEach { SpeechLanguages.requireSupported(it) }
    }
    @Test(expected = IllegalArgumentException::class) fun systemIsAClientPreferenceNotAModelLanguage() { SpeechLanguages.requireSupported("system") }
    @Test(expected = IllegalArgumentException::class) fun unsupportedLanguageIsRejected() { SpeechLanguages.requireSupported("fr") }
    @Test(expected = IllegalArgumentException::class) fun emptyIsNotAnImplicitAuto() { SpeechLanguages.requireSupported("") }
}
