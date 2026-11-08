package io.github.taehunkim.ondolsaem

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.engine.EmbeddingGemma2Embedder
import io.github.taehunkim.ondolsaem.engine.Npu
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.ByteArrayOutputStream
import java.io.File

/**
 * On a phone with an NPU build downloaded (OndolSaem › Models › "Download NPU build"): the NPU build loads on the
 * NPU and gives about the same vectors as the GPU/CPU build, so clients keep their index. Logs the cosine per input
 * and the time per embedding (tag OndolSaem.NpuTest). Skipped on phones without an NPU build (the emulator).
 *
 * Run without uninstalling the app (its models stay): `scripts/npu-check.sh`.
 */
@RunWith(AndroidJUnit4::class)
class NpuParityTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val manager get() = (ctx.applicationContext as OndolSaemApp).container.modelManager
    private val cache get() = File(ctx.cacheDir, "litertlm")

    private val texts = listOf(
        "오늘 점심값 12,000원 카드 영수증. 김밥천국 강남점.",
        "다음 주 제주도 3박 4일 일정: 첫날 성산일출봉, 둘째 날 한라산 등반.",
        "Quarterly sales report draft due Friday.",
        "고양이가 소파 위에서 자고 있다",
    )

    @Test fun embedding740m() = check("eg2-740m")

    private fun check(id: String) = runBlocking {
        val spec = ModelCatalog.byId(id)!!
        val device = Npu.device
        Log.i(TAG, "chip ${android.os.Build.SOC_MANUFACTURER} ${android.os.Build.SOC_MODEL} -> $device")
        val npu = manager.npuTarget(spec)
        assumeTrue("no NPU build of $id on this phone", npu != null && manager.isReady(spec))
        val file = manager.file(spec)!!
        val ref = EmbeddingGemma2Embedder.load(id, file, cache, lowPower = true) // CPU, the same as on every phone
        val gpu = EmbeddingGemma2Embedder.load(id, file, cache)
        val onNpu = EmbeddingGemma2Embedder.load(id, file, cache, npu = npu)
        try {
            assertEquals("$id did not load on the NPU", "NPU", onNpu.backend)
            val worst = ArrayList<Float>()
            val worstGpu = ArrayList<Float>()
            suspend fun compare(what: String, f: suspend (EmbeddingGemma2Embedder) -> FloatArray) {
                val a = f(ref)
                val b = f(gpu)
                val t = System.nanoTime()
                val c = f(onNpu)
                val ms = (System.nanoTime() - t) / 1_000_000
                val tg = System.nanoTime(); f(gpu); val msGpu = (System.nanoTime() - tg) / 1_000_000
                val cNpu = cos(a, c); val cGpu = cos(a, b)
                Log.i(TAG, "$id $what: cos(cpu,npu)=%.4f cos(cpu,gpu)=%.4f npu %d ms, gpu %d ms".format(cNpu, cGpu, ms, msGpu))
                worst.add(cNpu); worstGpu.add(cGpu)
            }
            for (t in texts) {
                compare("doc '${t.take(12)}'") { it.embedDocument(t, null) }
                compare("query '${t.take(12)}'") { it.embedQuery(t) }
            }
            compare("image") { it.embedImage(testImage()) }
            Log.i(TAG, "$id worst cos(cpu,npu)=%.4f, worst cos(cpu,gpu)=%.4f, backend after the run: ${onNpu.backend}".format(worst.min(), worstGpu.min()))
            assertTrue("$id: NPU vectors identical to the CPU's: the NPU did not run", worst.min() < 0.99999f)
            assertTrue("$id: NPU vectors differ (worst cos ${worst.min()})", worst.min() >= minOf(MIN_COS, worstGpu.min() - 0.01f))
        } finally {
            ref.close(); gpu.close(); onNpu.close()
        }
    }

    private fun cos(a: FloatArray, b: FloatArray): Float {
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return (d / Math.sqrt(na * nb)).toFloat()
    }

    private fun testImage(): ByteArray {
        val bmp = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
        Canvas(bmp).apply {
            drawColor(Color.rgb(250, 240, 220))
            drawCircle(256f, 256f, 160f, Paint().apply { color = Color.rgb(220, 60, 40) })
            drawText("EXIT 12", 180f, 470f, Paint().apply { color = Color.BLACK; textSize = 48f })
        }
        return ByteArrayOutputStream().also { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) }.toByteArray()
    }

    private companion object {
        const val TAG = "OndolSaem.NpuTest"
        /** Same model, other compiler and quantization: clients keep their index above this, or as close as the GPU is. */
        const val MIN_COS = 0.98f
    }
}
