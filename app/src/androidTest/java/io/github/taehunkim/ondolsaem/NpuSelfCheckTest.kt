package io.github.taehunkim.ondolsaem

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.engine.NpuTarget
import io.github.taehunkim.ondolsaem.runtime.NpuSelfCheck
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The self-check's report on a phone (or emulator) without a working NPU: backends that fail are reported with their
 * error, the CPU still runs every input, and the whole report is produced. Skipped without the 440M model.
 */
@RunWith(AndroidJUnit4::class)
class NpuSelfCheckTest {
    private val ctx = InstrumentationRegistry.getInstrumentation().targetContext
    private val manager get() = (ctx.applicationContext as OndolSaemApp).container.modelManager

    @Test fun failingBackendsAreReportedNotThrown() = runBlocking {
        val spec = ModelCatalog.byId("eg2-740m")!!
        assumeTrue("eg2-740m not downloaded", manager.isReady(spec))
        val file = manager.file(spec)!!
        val out = StringBuilder()
        // the main file as a stand-in "NPU build": loading it on the NPU fails here
        NpuSelfCheck(ctx, manager).checkEmbedding(spec.id, file, NpuTarget(file, ctx.applicationInfo.nativeLibraryDir)) { out.append(it).append('\n') }
        val report = out.toString()
        Log.i("OndolSaem.NpuTest", report)
        assertTrue(report, report.contains("CPU: loaded"))
        assertTrue(report, Regex("long doc \\d+ chars: \\d+ ms first").containsMatchIn(report))
        // the stand-in file has no NPU-compiled parts, so it runs on the CPU: the check must flag it, not call it a success
        assertTrue(report, report.contains("NPU: load failed") || report.contains("IDENTICAL to the CPU"))
    }

    @Test fun fullReportOnThisDevice() = runBlocking {
        val text = NpuSelfCheck(ctx, manager).run(useNpu = true) {}
        Log.i("OndolSaem.NpuTest", text)
        assertTrue(text, text.startsWith("# OndolSaem NPU check") && text.contains("chip "))
    }
}
