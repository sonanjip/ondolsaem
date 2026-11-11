package io.github.taehunkim.ondolsaem

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelSource
import io.github.taehunkim.ondolsaem.catalog.ModelSpec
import io.github.taehunkim.ondolsaem.download.DownloadException
import io.github.taehunkim.ondolsaem.download.ModelDownloader
import io.github.taehunkim.ondolsaem.download.ModelManager
import io.github.taehunkim.ondolsaem.download.ModelState
import io.github.taehunkim.ondolsaem.engine.EmbedTableQ8Converter
import io.github.taehunkim.ondolsaem.engine.TfliteW8Converter
import io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories
import io.github.taehunkim.ondolsaem.settings.HostSettings
import kotlinx.coroutines.runBlocking
import okhttp3.Interceptor
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Protocol
import okhttp3.Response
import okhttp3.ResponseBody.Companion.toResponseBody
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest

@RunWith(AndroidJUnit4::class)
class ModelConversionTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private lateinit var modelsDir: File
    private lateinit var manager: ModelManager

    @Before fun setUp() {
        modelsDir = File(ctx.cacheDir, "model_conversion_test_${System.nanoTime()}").apply { mkdirs() }
        manager = ModelManager(ctx, HostSettings(ctx), modelsDir = modelsDir)
    }

    @After fun tearDown() {
        modelsDir.deleteRecursively()
    }

    @Test fun managerRecognizesConvertedAndLegacyD1OmniFiles() {
        val spec = ModelCatalog.byId("d1-omni-2b")!!
        val dir = manager.dir(spec).apply { mkdirs() }
        assertFalse(manager.isReady(spec))

        // Legacy v0.7.1 raw layout
        File(dir, "d1-omni-2b_decide_L512_fp16fc.tflite").writeBytes(ByteArray(4))
        File(dir, "d1-omni-2b_tower_G24x24_fp16fc.tflite").writeBytes(ByteArray(4))
        File(dir, "embed_table.npy").writeBytes(ByteArray(4))
        File(dir, "tokenizer.json").writeBytes(ByteArray(4))
        assertTrue(manager.isReady(spec))
        assertEquals("d1-omni-2b_decide_L512_fp16fc.tflite", manager.modelFile(spec)!!.name)

        // Converted layout replaces raw files except tokenizer.json
        File(dir, "d1-omni-2b_decide_L512_fp16fc.tflite").delete()
        File(dir, "d1-omni-2b_tower_G24x24_fp16fc.tflite").delete()
        File(dir, "embed_table.npy").delete()
        assertFalse(manager.isReady(spec))

        File(dir, "d1-omni-2b_decide_L512_i8w.tflite").writeBytes(ByteArray(4))
        File(dir, "d1-omni-2b_tower_G24x24_i8w.tflite").writeBytes(ByteArray(4))
        File(dir, "embed_table.q8").writeBytes(ByteArray(4))
        assertTrue(manager.isReady(spec))
        assertEquals("d1-omni-2b_decide_L512_i8w.tflite", manager.modelFile(spec)!!.name)
    }

    @Test fun managerRecognizesConvertedIrisFiles() {
        val spec = ModelCatalog.byId("iris-3b-turbo")!!
        val src = spec.source as ModelSource.HuggingFace
        val dir = manager.dir(spec).apply { mkdirs() }
        assertFalse(manager.isReady(spec))

        File(dir, src.fileName.substringAfterLast('/')).writeBytes(ByteArray(4))
        for (path in src.extras.keys) {
            val rawName = path.substringAfterLast('/')
            val conv = src.extraConversions[path]
            val targetName = if (conv != null) ModelSource.Conversions.convertedFileName(rawName, conv) else rawName
            File(dir, targetName).writeBytes(ByteArray(4))
        }
        assertTrue(manager.isReady(spec))
    }

    @Test fun convertedTfliteRunsInLiteRtCompiledModelOnAndroid() {
        val src = File(modelsDir, "tiny_fp16fc.tflite")
        val dst = File(modelsDir, "tiny_i8w.tflite")
        // Identity-like 4x4 weight matrix in FP16: diag(1, -2, 0.5, 4)
        val fp16 = shortArrayOf(
            0x3c00, 0x0000, 0x0000, 0x0000,
            0x0000, 0xc000.toShort(), 0x0000, 0x0000,
            0x0000, 0x0000, 0x3800, 0x0000,
            0x0000, 0x0000, 0x0000, 0x4400,
        )
        val wBytes = ByteBuffer.allocate(fp16.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            .apply { for (s in fp16) putShort(s) }.array()
        writeTinyDequantFcModel(src, 4, 4, wBytes)

        val res = TfliteW8Converter.convert(src, dst)
        assertEquals(1, res.quantizedWeights)

        val env = Environment.create(ctx)
        try {
            val model = CompiledModel.create(dst.absolutePath, CompiledModel.Options(Accelerator.CPU), env)
            try {
                val inBuf = model.createInputBuffer("x", "serving_default")
                val outBuf = model.createOutputBuffer("y", "serving_default")
                inBuf.writeFloat(floatArrayOf(2f, 3f, 4f, -1f))
                model.run(mapOf("x" to inBuf), mapOf("y" to outBuf), "serving_default")
                val got = outBuf.readFloat()
                inBuf.close()
                outBuf.close()
                val expected = floatArrayOf(2f, -6f, 2f, -4f)
                for (i in expected.indices) assertEquals(expected[i], got[i], 0.05f)
            } finally {
                model.close()
            }
        } finally {
            env.close()
        }
    }

    @Test fun runDownloadConvertsMainAndExtrasAndSkipsConvertedOnRetry() = runBlocking {
        val rawTflite = File(modelsDir, "payload_fp16fc.tflite").apply {
            val wBytes = ByteBuffer.allocate(16 * 2).order(ByteOrder.LITTLE_ENDIAN)
                .apply { repeat(16) { putShort(0x3c00) } }.array()
            writeTinyDequantFcModel(this, 4, 4, wBytes)
        }.readBytes()
        val rawNpy = makeNpyBf16(rows = 2, cols = 4, values = floatArrayOf(1f, -2f, 0.5f, 0f, 0f, 0f, 0f, 0f))
        val tokBytes = """{"model":{"vocab":{}}}""".toByteArray()

        fun sha256(b: ByteArray): String =
            MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }

        val spec = ModelSpec(
            id = "test-d1-conv",
            category = Categories.DECISION,
            displayName = "Test",
            descriptionRes = R.string.model_d1_omni_desc,
            sizeBytes = (rawTflite.size + rawNpy.size + tokBytes.size).toLong(),
            license = "Apache-2.0",
            source = ModelSource.HuggingFace(
                repo = "example/test",
                fileName = "decide_fp16fc.tflite",
                sha256 = sha256(rawTflite),
                extras = mapOf(
                    "tables/embed_table.npy" to sha256(rawNpy),
                    "provider/tokenizer.json" to sha256(tokBytes),
                ),
                convert = ModelSource.Conversions.TFLITE_W8,
                extraConversions = mapOf(
                    "tables/embed_table.npy" to ModelSource.Conversions.EMBED_NPY_Q8,
                ),
            ),
        )

        var failTokenizer = true
        val requests = ArrayList<String>()
        val client = OkHttpClient.Builder().addInterceptor(Interceptor { chain ->
            val req = chain.request()
            val path = req.url.encodedPath
            requests.add(path)
            val (code, body) = when {
                path.endsWith("/decide_fp16fc.tflite") -> 200 to rawTflite
                path.endsWith("/tables/embed_table.npy") -> 200 to rawNpy
                path.endsWith("/provider/tokenizer.json") && failTokenizer -> 500 to ByteArray(0)
                path.endsWith("/provider/tokenizer.json") -> 200 to tokBytes
                else -> 404 to ByteArray(0)
            }
            Response.Builder()
                .request(req)
                .protocol(Protocol.HTTP_1_1)
                .code(code)
                .message(if (code == 200) "OK" else "ERR")
                .body(body.toResponseBody("application/octet-stream".toMediaType()))
                .build()
        }).build()

        val dlManager = ModelManager(ctx, HostSettings(ctx), ModelDownloader(client), modelsDir)
        try {
            dlManager.runDownload(spec) { _, _ -> }
            fail("expected tokenizer HTTP 500")
        } catch (e: DownloadException) {
            assertEquals(DownloadException.Kind.HTTP, e.kind)
        }
        assertFalse(dlManager.isReady(spec))
        assertTrue(dlManager.states.value[spec.id] is ModelState.Failed)
        val dir = dlManager.dir(spec)
        // Converted outputs already exist and raw downloads were deleted.
        assertTrue(File(dir, "decide_i8w.tflite").exists())
        assertFalse(File(dir, "decide_fp16fc.tflite").exists())
        assertTrue(EmbedTableQ8Converter.isQ8File(File(dir, "embed_table.q8")))
        assertFalse(File(dir, "embed_table.npy").exists())

        // Retry: must only request tokenizer.json and must not re-download converted files.
        requests.clear()
        failTokenizer = false
        dlManager.runDownload(spec) { _, _ -> }
        assertEquals(listOf("/example/test/resolve/main/provider/tokenizer.json"), requests)
        assertEquals(ModelState.Installed, dlManager.states.value[spec.id])
        assertTrue(dlManager.isReady(spec))
        assertEquals("decide_i8w.tflite", dlManager.modelFile(spec)!!.name)
    }

    private fun makeNpyBf16(rows: Int, cols: Int, values: FloatArray): ByteArray {
        val rawDict = "{'descr': '<u2', 'fortran_order': False, 'shape': ($rows, $cols), }"
        val pad = (16 - (10 + rawDict.length + 1) % 16) % 16
        val dict = rawDict + " ".repeat(pad) + "\n"
        val hdr = ByteArray(10 + dict.length)
        hdr[0] = 0x93.toByte()
        "NUMPY".toByteArray(Charsets.US_ASCII).copyInto(hdr, 1)
        hdr[6] = 1
        hdr[8] = (dict.length and 0xff).toByte()
        hdr[9] = ((dict.length ushr 8) and 0xff).toByte()
        dict.toByteArray(Charsets.US_ASCII).copyInto(hdr, 10)
        val data = ByteBuffer.allocate(values.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in values) data.putShort((java.lang.Float.floatToRawIntBits(v) ushr 16).toShort())
        return hdr + data.array()
    }

    private fun writeTinyDequantFcModel(dst: File, outCh: Int, inCh: Int, wFp16: ByteArray) {
        val fb = SmallFb()
        val b0 = fb.table(0) {}
        val b1 = fb.table(1) { addOff(0, fb.byteVec(wFp16)) }
        val bufs = fb.offVec(intArrayOf(b0, b1))

        val ocDq = fb.table(4) { addByte(0, 6); addInt(2, 1); addInt(3, 6) }
        val ocFc = fb.table(4) { addByte(0, 9); addInt(2, 5); addInt(3, 9) }
        val ocs = fb.offVec(intArrayOf(ocDq, ocFc))

        fun tensor(name: String, shape: IntArray, type: Byte, buf: Int): Int {
            val s = fb.intVec(shape); val n = fb.str(name)
            return fb.table(9) { addOff(0, s); addByte(1, type); if (buf != 0) addInt(2, buf); addOff(3, n); addByte(8, 1) }
        }
        val t0 = tensor("x", intArrayOf(1, inCh), 0, 0)
        val t1 = tensor("w16", intArrayOf(outCh, inCh), 1, 1)
        val t2 = tensor("w_dq", intArrayOf(outCh, inCh), 0, 0)
        val t3 = tensor("y", intArrayOf(1, outCh), 0, 0)
        val tensors = fb.offVec(intArrayOf(t0, t1, t2, t3))

        val opDq = fb.table(3) { addInt(0, 0); addOff(1, fb.intVec(intArrayOf(1))); addOff(2, fb.intVec(intArrayOf(2))) }
        val fcOpts = fb.table(4) { addByte(2, 1) }
        val opFc = fb.table(5) {
            addInt(0, 1); addOff(1, fb.intVec(intArrayOf(0, 2))); addOff(2, fb.intVec(intArrayOf(3)))
            addByte(3, 8); addOff(4, fcOpts)
        }
        val ops = fb.offVec(intArrayOf(opDq, opFc))

        val sg = fb.table(4) {
            addOff(0, tensors); addOff(1, fb.intVec(intArrayOf(0))); addOff(2, fb.intVec(intArrayOf(3))); addOff(3, ops)
        }
        val sd = fb.table(5) {
            addOff(0, fb.offVec(intArrayOf(fb.table(2) { addOff(0, fb.str("x")) })))
            addOff(1, fb.offVec(intArrayOf(fb.table(2) { addOff(0, fb.str("y")); addInt(1, 3) })))
            addOff(2, fb.str("serving_default"))
            addInt(4, 0)
        }
        val model = fb.table(8) {
            addInt(0, 3); addOff(1, ocs); addOff(2, fb.offVec(intArrayOf(sg))); addOff(4, bufs); addOff(7, fb.offVec(intArrayOf(sd)))
        }
        dst.writeBytes(fb.finish(model))
    }

    private class SmallFb {
        private var buf = ByteArray(16 * 1024)
        private var space = buf.size
        private var slots = IntArray(0)
        private var bottom = 0
        fun off() = buf.size - space
        private fun ensure(n: Int) {
            if (space >= n) return
            var cap = buf.size; val used = off(); while (cap - used < n) cap = cap shl 1
            val next = ByteArray(cap); System.arraycopy(buf, space, next, cap - used, used); buf = next; space = cap - used
        }
        private fun prep(align: Int, add: Int) {
            val m = (off() + add) % align; val p = if (m == 0) 0 else align - m; ensure(p); repeat(p) { buf[--space] = 0 }
        }
        fun putByte(v: Byte) { ensure(1); buf[--space] = v }
        fun putShort(v: Short) { prep(2, 0); ensure(2); buf[--space] = (v.toInt() ushr 8).toByte(); buf[--space] = v.toByte() }
        fun putInt(v: Int) { prep(4, 0); ensure(4); for (k in 3 downTo 0) buf[--space] = (v ushr (k * 8)).toByte() }
        fun putBytes(b: ByteArray) { ensure(b.size); space -= b.size; System.arraycopy(b, 0, buf, space, b.size) }
        fun putOff(t: Int) { prep(4, 0); putInt((off() + 4) - t) }
        fun byteVec(b: ByteArray): Int { prep(16, b.size); putBytes(b); putInt(b.size); return off() }
        fun str(s: String): Int { val b = s.toByteArray(); prep(4, b.size + 1); putByte(0); putBytes(b); putInt(b.size); return off() }
        fun intVec(a: IntArray): Int { prep(4, a.size * 4); for (i in a.size - 1 downTo 0) putInt(a[i]); putInt(a.size); return off() }
        fun offVec(a: IntArray): Int { prep(4, a.size * 4); for (i in a.size - 1 downTo 0) putOff(a[i]); putInt(a.size); return off() }
        fun addByte(i: Int, v: Byte) { putByte(v); slots[i] = off() }
        fun addInt(i: Int, v: Int) { putInt(v); slots[i] = off() }
        fun addOff(i: Int, t: Int) { putOff(t); slots[i] = off() }
        fun table(n: Int, block: SmallFb.() -> Unit): Int {
            val prevSlots = slots; val prevBottom = bottom
            slots = IntArray(n); bottom = off(); block(); putInt(0); val ts = off(); val os = ts - bottom
            for (i in n - 1 downTo 0) putShort((if (slots[i] == 0) 0 else ts - slots[i]).toShort())
            putShort(os.toShort()); putShort((4 + n * 2).toShort())
            val vo = off(); val sp = buf.size - ts; val soff = vo - ts
            for (k in 0..3) buf[sp + k] = (soff ushr (k * 8)).toByte()
            slots = prevSlots; bottom = prevBottom
            return ts
        }
        fun finish(root: Int): ByteArray { prep(16, 8); putBytes("TFL3".toByteArray()); putOff(root); return buf.copyOfRange(space, buf.size) }
    }
}
