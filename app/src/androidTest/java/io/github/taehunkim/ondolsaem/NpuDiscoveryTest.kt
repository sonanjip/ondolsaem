package io.github.taehunkim.ondolsaem

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.NpuDiscovery
import io.github.taehunkim.ondolsaem.download.DownloadException
import io.github.taehunkim.ondolsaem.engine.Npu
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeNoException
import org.junit.Test
import org.junit.runner.RunWith

/** Searches the real Hugging Face repositories for a few chips (needs the network; skipped offline). */
@RunWith(AndroidJUnit4::class)
class NpuDiscoveryTest {
    private fun check(soc: String): NpuDiscovery.Result = runBlocking {
        try {
            NpuDiscovery().check(ModelCatalog.all, Npu.parse(soc)!!, null, System.currentTimeMillis())
        } catch (e: DownloadException) {
            assumeNoException("offline?", e); throw e
        }.also { r -> Log.i("OndolSaem.NpuTest", "$soc: ${r.builds.mapValues { "${it.value.fileName} ${it.value.sizeBytes}" }}") }
    }

    @Test fun snapdragon8EliteHasEmbeddingBuilds() {
        val r = check("SM8750")
        assertEquals(setOf("eg2-740m"), r.builds.keys)
        assertTrue(r.builds.values.all { it.fileName.endsWith("_Qualcomm_SM8750.litertlm") && it.sha256.length == 64 })
        assertTrue("chat models were checked too", "chat-gemma4-e4b" in r.checked)
    }

    @Test fun tensorG5HasEmbeddingAndChatBuilds() {
        assertTrue("chat-gemma4-e2b" in check("Tensor G5").builds.keys)
    }

    @Test fun anOlderChipHasNone() {
        assertTrue(check("SM8450").builds.isEmpty())
    }
}
