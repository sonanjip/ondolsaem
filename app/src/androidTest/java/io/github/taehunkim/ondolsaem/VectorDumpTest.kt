package io.github.taehunkim.ondolsaem

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.engine.EmbeddingGemma2Embedder
import kotlinx.coroutines.runBlocking
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Writes CPU vectors of fixed texts to the app's cache (vec-<tag>.txt), to compare runtime versions. Run with -e dumpTag <tag>. */
@RunWith(AndroidJUnit4::class)
class VectorDumpTest {
    @Test fun dump() = runBlocking {
        val tag = InstrumentationRegistry.getArguments().getString("dumpTag")
        assumeTrue("no dumpTag", tag != null)
        val ctx = InstrumentationRegistry.getInstrumentation().targetContext
        val manager = (ctx.applicationContext as OndolSaemApp).container.modelManager
        val out = StringBuilder()
        for (id in listOf("eg2-740m")) {
            val spec = ModelCatalog.byId(id)!!
            if (!manager.isReady(spec)) continue
            EmbeddingGemma2Embedder.load(id, manager.file(spec)!!, File(ctx.cacheDir, "litertlm")).use { e ->
                for (t in listOf("오늘 점심값 12,000원 카드 영수증.", "Quarterly sales report draft due Friday.", "고양이가 소파 위에서 자고 있다")) {
                    out.append(id).append('\t').append(t).append('\t').append(e.embedDocument(t, null).joinToString(",")).append('\n')
                }
            }
        }
        File(ctx.cacheDir, "vec-$tag.txt").writeText(out.toString())
    }
}
