package io.github.taehunkim.ondolsaem.download

import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelSource

import org.apache.commons.compress.archivers.tar.TarArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveOutputStream
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorOutputStream
import org.junit.After
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import java.io.File
import java.io.IOException
import java.nio.file.Files

class ArchiveExtractorTest {
    private lateinit var dir: File

    @Before fun setUp() { dir = Files.createTempDirectory("arc").toFile() }
    @After fun tearDown() { dir.deleteRecursively() }

    private fun archive(vararg entries: Pair<String, ByteArray>): File {
        val f = File(dir, "a.tar.bz2")
        TarArchiveOutputStream(BZip2CompressorOutputStream(f.outputStream())).use { t ->
            for ((name, data) in entries) {
                t.putArchiveEntry(TarArchiveEntry(name).apply { size = data.size.toLong() }); t.write(data); t.closeArchiveEntry()
            }
        }
        return f
    }

    @Test fun extractsOnlyTheWantedFilesFromTheNestedFolder() {
        val model = ByteArray(300_000) { (it % 251).toByte() }
        val a = archive("top/model.int8.onnx" to model, "top/tokens.txt" to "a\nb".toByteArray(), "top/README.md" to ByteArray(10), "top/test_wavs/0.wav" to ByteArray(5000))
        val out = File(dir, "out")
        ArchiveExtractor.extract(a, out, mapOf("model.int8.onnx" to "model.int8.onnx", "tokens.txt" to "tokens.txt"))
        assertArrayEquals(model, File(out, "model.int8.onnx").readBytes())
        assertEquals("a\nb", File(out, "tokens.txt").readText())
        assertEquals(setOf("model.int8.onnx", "tokens.txt"), out.list()!!.toSet())
    }

    @Test fun suffixMatchingDoesNotConfuseSimilarNames() {
        val a = archive("t/not_tokens.txt" to "wrong".toByteArray(), "t/tokens.txt" to "right".toByteArray())
        val out = File(dir, "out")
        ArchiveExtractor.extract(a, out, mapOf("tokens.txt" to "tokens.txt"))
        assertEquals("right", File(out, "tokens.txt").readText())
    }

    @Test fun localNamesCanDifferFromEntryNames() {
        val a = archive("x/encoder_model.ort" to "enc".toByteArray())
        val out = File(dir, "out")
        ArchiveExtractor.extract(a, out, mapOf("encoder_model.ort" to "encoder.ort"))
        assertEquals("enc", File(out, "encoder.ort").readText())
    }

    @Test fun missingFileIsAnError() {
        val a = archive("t/tokens.txt" to "x".toByteArray())
        try { ArchiveExtractor.extract(a, File(dir, "out"), mapOf("tokens.txt" to "tokens.txt", "model.int8.onnx" to "model.int8.onnx")); fail() }
        catch (e: IOException) { assertTrue(e.message!!, e.message!!.contains("model.int8.onnx")) }
    }

    @Test fun noTemporaryFilesRemainAndReextractionOverwrites() {
        val out = File(dir, "out")
        ArchiveExtractor.extract(archive("t/tokens.txt" to "v1".toByteArray()), out, mapOf("tokens.txt" to "tokens.txt"))
        ArchiveExtractor.extract(archive("t/tokens.txt" to "v2".toByteArray()), out, mapOf("tokens.txt" to "tokens.txt"))
        assertEquals("v2", File(out, "tokens.txt").readText())
        assertFalse(out.list()!!.any { it.endsWith(".part") })
    }

    @Test fun corruptArchiveFailsCleanly() {
        val f = File(dir, "bad.tar.bz2").also { it.writeText("not bzip2 at all") }
        try { ArchiveExtractor.extract(f, File(dir, "out"), mapOf("tokens.txt" to "tokens.txt")); fail() } catch (_: IOException) { }
    }

    @Test fun catalogEntryForSenseVoiceUsesTheReleaseArchive() {
        val s = ModelCatalog.byId("asr-sensevoice")!!.source as ModelSource.Archive
        assertTrue(s.url.endsWith(".tar.bz2") && s.url.contains("sense-voice"))
        assertEquals(setOf("model.int8.onnx", "tokens.txt"), s.files.keys)
        assertEquals("asr", ModelCatalog.byId("asr-sensevoice")!!.category)
        assertTrue(ModelCatalog.byId("eg2-740m")!!.audio)
    }
}
