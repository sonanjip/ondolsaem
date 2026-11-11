package io.github.taehunkim.ondolsaem.download

import com.google.mlkit.genai.common.DownloadStatus
import com.google.mlkit.genai.common.FeatureStatus
import com.google.mlkit.genai.common.GenAiException
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.engine.AiCoreGateway
import io.github.taehunkim.ondolsaem.engine.NanoChatSession
import io.github.taehunkim.ondolsaem.runtime.ModelRuntime
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class ModelManagerAiCoreTest {

    private class FakeGateway(
        private val status: Int,
        private val statuses: List<DownloadStatus> = emptyList(),
    ) : AiCoreGateway {
        override suspend fun checkStatus(): Int = status
        override fun download(): Flow<DownloadStatus> = flow { statuses.forEach { emit(it) } }
        override suspend fun openChatSession(): NanoChatSession = throw UnsupportedOperationException()
    }

    @Test fun mapsAiCoreFeatureStatusesToModelState() {
        assertEquals(ModelState.Unavailable, ModelManager.aiCoreStateFor(null, null))
        assertEquals(ModelState.Unavailable, ModelManager.aiCoreStateFor(FeatureStatus.UNAVAILABLE, null))
        assertEquals(ModelState.NotInstalled, ModelManager.aiCoreStateFor(FeatureStatus.DOWNLOADABLE, null))
        assertEquals(ModelState.Downloading(0, 0), ModelManager.aiCoreStateFor(FeatureStatus.DOWNLOADING, null))
        assertEquals(ModelState.Installed, ModelManager.aiCoreStateFor(FeatureStatus.AVAILABLE, null))

        val failed = ModelState.Failed(DownloadException.Kind.HTTP, "oops")
        assertEquals(failed, ModelManager.aiCoreStateFor(FeatureStatus.DOWNLOADABLE, failed))
    }

    @Test fun streamsDownloadProgressAndCompletes() = runBlocking {
        val gw = FakeGateway(
            status = FeatureStatus.DOWNLOADABLE,
            statuses = listOf(
                DownloadStatus.DownloadStarted(1000L),
                DownloadStatus.DownloadProgress(400L),
                DownloadStatus.DownloadProgress(1000L),
                DownloadStatus.DownloadCompleted,
            ),
        )
        val states = mutableListOf<ModelState>()
        val finalStatus = ModelManager.runAiCoreDownload(gw) { states += it }
        assertEquals(FeatureStatus.AVAILABLE, finalStatus)
        assertEquals(
            listOf(
                ModelState.Downloading(0, 1000L),
                ModelState.Downloading(400L, 1000L),
                ModelState.Downloading(1000L, 1000L),
            ),
            states,
        )
    }

    @Test fun failsWhenAiCoreUnavailableOrDownloadFails() = runBlocking {
        try {
            ModelManager.runAiCoreDownload(FakeGateway(FeatureStatus.UNAVAILABLE)) {}
            fail("expected NOT_FOUND")
        } catch (e: DownloadException) {
            assertEquals(DownloadException.Kind.NOT_FOUND, e.kind)
        }

        val failedGw = FakeGateway(
            status = FeatureStatus.DOWNLOADABLE,
            statuses = listOf(
                DownloadStatus.DownloadStarted(500L),
                DownloadStatus.DownloadFailed(GenAiException("disk full", null, GenAiException.ErrorCode.NOT_ENOUGH_DISK_SPACE)),
            ),
        )
        try {
            ModelManager.runAiCoreDownload(failedGw) {}
            fail("expected HTTP")
        } catch (e: DownloadException) {
            assertEquals(DownloadException.Kind.HTTP, e.kind)
            assertTrue(e.message!!.contains("disk full"))
        }
    }

    @Test fun capsGeminiNanoContextWindowTo4096() {
        val nano = ModelCatalog.byId("chat-gemini-nano")!!
        val gemma = ModelCatalog.byId("chat-gemma4-e2b")!!
        assertEquals(2048, ModelRuntime.effectiveContextTokens(nano, 2048))
        assertEquals(4096, ModelRuntime.effectiveContextTokens(nano, 4096))
        assertEquals(4096, ModelRuntime.effectiveContextTokens(nano, 8192))
        assertEquals(8192, ModelRuntime.effectiveContextTokens(gemma, 8192))
    }
}
