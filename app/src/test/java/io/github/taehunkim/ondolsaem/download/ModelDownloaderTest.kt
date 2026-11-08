package io.github.taehunkim.ondolsaem.download

import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelSource

import kotlinx.coroutines.runBlocking
import okhttp3.mockwebserver.MockResponse
import okhttp3.mockwebserver.MockWebServer
import okio.Buffer
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import java.io.File
import java.nio.file.Files

class ModelDownloaderTest {
    private lateinit var server: MockWebServer
    private lateinit var dir: File
    private val payload = ByteArray(300_000) { (it % 251).toByte() }

    @Before fun setUp() { server = MockWebServer().also { it.start() }; dir = Files.createTempDirectory("dl").toFile() }
    @After fun tearDown() { server.shutdown(); dir.deleteRecursively() }

    private fun body(bytes: ByteArray) = Buffer().write(bytes)

    @Test fun downloadsFullFileAndSendsToken() = runBlocking {
        server.enqueue(MockResponse().setBody(body(payload)))
        val target = File(dir, "m.bin")
        ModelDownloader().download(server.url("/m").toString(), target, "tok") { _, _ -> }
        assertArrayEquals(payload, target.readBytes())
        assertFalse(File(target.path + ".part").exists())
        assertEquals("Bearer tok", server.takeRequest().getHeader("Authorization"))
    }

    @Test fun resumesFromPartFile() = runBlocking {
        val target = File(dir, "m.bin")
        File(target.path + ".part").writeBytes(payload.copyOfRange(0, 100_000))
        server.enqueue(
            MockResponse().setResponseCode(206)
                .setHeader("Content-Range", "bytes 100000-299999/300000")
                .setBody(body(payload.copyOfRange(100_000, payload.size))),
        )
        ModelDownloader().download(server.url("/m").toString(), target, null) { _, _ -> }
        assertArrayEquals(payload, target.readBytes())
        assertEquals("bytes=100000-", server.takeRequest().getHeader("Range"))
    }

    @Test fun serverIgnoringRangeRestartsFromScratch() = runBlocking {
        val target = File(dir, "m.bin")
        File(target.path + ".part").writeBytes(ByteArray(5000) { 9 })
        server.enqueue(MockResponse().setBody(body(payload)))
        ModelDownloader().download(server.url("/m").toString(), target, null) { _, _ -> }
        assertArrayEquals(payload, target.readBytes())
    }

    @Test fun authAndNotFoundAreClassified() = runBlocking {
        server.enqueue(MockResponse().setResponseCode(401))
        server.enqueue(MockResponse().setResponseCode(404))
        val kinds = (1..2).map {
            try { ModelDownloader().download(server.url("/m").toString(), File(dir, "x"), null) { _, _ -> }; null }
            catch (e: DownloadException) { e.kind }
        }
        assertEquals(listOf(DownloadException.Kind.AUTH, DownloadException.Kind.NOT_FOUND), kinds)
    }

    @Test fun catalogUrlIsWellFormed() {
        val s = ModelCatalog.byId("eg2-740m")!!.source as ModelSource.HuggingFace
        assertTrue(s.url.startsWith("https://huggingface.co/litert-community/") && s.url.endsWith(".litertlm"))
    }
}
