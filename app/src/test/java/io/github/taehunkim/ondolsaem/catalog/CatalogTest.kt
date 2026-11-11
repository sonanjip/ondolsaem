package io.github.taehunkim.ondolsaem.catalog

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CatalogTest {
    @Test fun everyModelBelongsToAKnownCategory() {
        for (m in ModelCatalog.all) assertNotNull("${m.id} -> ${m.category}", ModelCategories.byId(m.category))
    }

    @Test fun categoryDefaultsExistInTheirCategory() {
        for (c in ModelCategories.all) {
            val spec = ModelCatalog.byId(c.defaultModelId)
            assertNotNull(c.id, spec)
            assertEquals(c.id, spec!!.category)
            assertTrue("default of ${c.id} must be usable", spec.source !is ModelSource.Planned)
        }
    }

    @Test fun idsAreUnique() {
        assertEquals(ModelCatalog.all.size, ModelCatalog.all.map { it.id }.toSet().size)
    }

    @Test fun embeddingChangeRequiresReindex() {
        assertEquals(ChangeImpact.REINDEX_REQUIRED, ModelCategories.byId("embedding")!!.changeImpact)
    }

    @Test fun everyDownloadHasAChecksum() {
        for (m in ModelCatalog.all) when (val s = m.source) {
            is ModelSource.HuggingFace -> {
                assertTrue(m.id, s.sha256?.length == 64)
                s.extras.forEach { (path, sha) -> assertTrue("${m.id} $path", sha?.length == 64) }
            }
            is ModelSource.Archive -> assertEquals(m.id, s.files.values.toSet(), s.sha256.keys)
            else -> Unit
        }
    }

    @Test fun decisionModelsBringTheirTokenizer() {
        for (id in listOf("nli-mdeberta-v3-w8", "nli-mdeberta-v3-q4")) {
            val s = ModelCatalog.byId(id)!!.source as ModelSource.HuggingFace
            assertTrue(id, s.extras.containsKey("tokenizer.json"))
        }
        val d1 = ModelCatalog.byId("d1-omni-2b")!!.source as ModelSource.HuggingFace
        assertTrue("d1-omni-2b", d1.extras.containsKey("provider/tokenizer.json"))
    }

    @Test fun geminiNanoUsesAiCoreInLlmCategory() {
        val nano = ModelCatalog.byId("chat-gemini-nano")!!
        assertEquals("llm", nano.category)
        assertTrue(nano.vision)
        assertTrue(nano.source is ModelSource.AiCore)
    }

    @Test fun conversionsHaveTheirTemplateInAssetsAndValidKeys() {
        for (m in ModelCatalog.all) {
            val s = m.source as? ModelSource.HuggingFace ?: continue
            s.convert?.let { c ->
                assertTrue("${m.id}: unknown convert $c", c in ModelSource.Conversions.ALL)
                if (c == ModelSource.Conversions.NLI_W8) {
                    for (f in listOf("model.onnx", "recipe.tsv")) {
                        assertTrue("${m.id}: $c/$f", java.io.File("src/main/assets/$c/$f").exists())
                    }
                }
            }
            s.extraConversions.forEach { (path, c) ->
                assertTrue("${m.id}: extraConversion $path not in extras", path in s.extras)
                assertTrue("${m.id}: unknown extraConversion $c", c in ModelSource.Conversions.ALL)
            }
        }
    }
}
