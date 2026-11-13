package io.github.taehunkim.ondolsaem

import androidx.test.platform.app.InstrumentationRegistry
import io.github.taehunkim.ondolsaem.sdk.OndolSaemClient
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/** Opt-in real-model comparison; fixtures are pushed into target files, never shipped with the app. */
class SpeechLanguageSmokeTest {
    @Test fun realSenseVoiceHonorsLanguageAndReturnsToAuto() = runBlocking {
        assumeTrue(InstrumentationRegistry.getArguments().getString("realAsr") == "true")
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val english = File(context.filesDir, "asr-language-en.pcm")
        val korean = File(context.filesDir, "asr-language-ko.pcm")
        assumeTrue(english.exists() && korean.exists())
        val client = OndolSaemClient(context)
        try {
            assertTrue(client.canTranscribe())
            suspend fun hear(file: File, language: String?): String {
                val lines = ArrayList<String>()
                val cue: suspend (Long, Long, String) -> Unit = { _, _, text -> lines += text }
                if (language == null) client.transcribe(file, 0, onCue = cue)
                else client.transcribeWithLanguage(file, 0, language, onCue = cue)
                return lines.joinToString(" ")
            }
            val before = hear(english, null)
            val en = hear(english, "en")
            val ko = hear(korean, "ko")
            val after = hear(english, null)
            assertTrue(en.isNotBlank())
            assertTrue(ko.isNotBlank())
            assertEquals("Explicit-language requests must not affect later automatic requests", before, after)
            File(context.filesDir, "asr-language-results.txt").writeText("auto: $before\nen: $en\nko: $ko\nauto-after: $after\n")
        } finally { client.disconnect() }
    }
}
