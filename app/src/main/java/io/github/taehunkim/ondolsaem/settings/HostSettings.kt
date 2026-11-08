package io.github.taehunkim.ondolsaem.settings

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.intPreferencesKey
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.core.stringSetPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.runtime.Residency
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.dataStore by preferencesDataStore("host_settings")

/** Host settings: the model and residency of each category, the Hugging Face token, known client apps. */
class HostSettings(private val context: Context) {
    private val store get() = context.dataStore

    private fun activeKey(category: String) = stringPreferencesKey("active_$category")
    private fun residencyKey(category: String) = stringPreferencesKey("residency_$category")
    private val hfTokenKey = stringPreferencesKey("hf_token")
    private val chatContextKey = intPreferencesKey("chat_context_tokens")
    private val clientsKey = stringSetPreferencesKey("clients")
    private val approvalsKey = stringPreferencesKey("approvals")
    private val lastVersionKey = intPreferencesKey("last_version_code")
    private val useNpuKey = androidx.datastore.preferences.core.booleanPreferencesKey("use_npu")
    private val onboardingKey = androidx.datastore.preferences.core.booleanPreferencesKey("onboarding_done")
    private fun clientCategoriesKey(pkg: String) = stringSetPreferencesKey("client_categories_$pkg")

    fun activeModelId(category: String): Flow<String> = store.data.map { prefs ->
        val saved = prefs[activeKey(category)]
        if (saved != null && ModelCatalog.byId(saved)?.category == category) {
            saved
        } else {
            ModelCategories.byId(category)?.defaultModelId.orEmpty()
        }
    }

    suspend fun setActiveModel(category: String, id: String) { store.edit { it[activeKey(category)] = id } }

    /** True when the user (or an upgrade step) stored a model for [category]; false = the category default applies. */
    suspend fun hasActiveModel(category: String): Boolean = store.data.first().contains(activeKey(category))

    /** Any model or memory choice stored (an install that was used before versions were recorded). */
    suspend fun hasStoredChoices(): Boolean = store.data.first().asMap().keys.any { it.name.startsWith("active_") || it.name.startsWith("residency_") }

    /** versionCode of the last run (0 = fresh install or a version before 0.4.0), see HostUpgrade. */
    val lastVersionCode: Flow<Int> = store.data.map { it[lastVersionKey] ?: 0 }
    suspend fun setLastVersionCode(v: Int) { store.edit { it[lastVersionKey] = v } }

    /** The first-run screens were completed (or skipped by an upgrade). */
    val onboardingDone: Flow<Boolean> = store.data.map { it[onboardingKey] ?: false }
    suspend fun setOnboardingDone(done: Boolean) { store.edit { it[onboardingKey] = done } }

    fun residency(category: String): Flow<Residency> = store.data.map { p ->
        p[residencyKey(category)]?.let { runCatching { Residency.valueOf(it) }.getOrNull() }
            ?: ModelCategories.byId(category)?.defaultResidency ?: Residency.IDLE_2M
    }

    suspend fun setResidency(category: String, r: Residency) { store.edit { it[residencyKey(category)] = r.name } }

    /** Third-party app decisions (service/Approvals.kt), as JSON. */
    val approvalsJson: Flow<String> = store.data.map { it[approvalsKey].orEmpty() }
    suspend fun setApprovalsJson(json: String) { store.edit { it[approvalsKey] = json } }

    val hfToken: Flow<String> = store.data.map { it[hfTokenKey].orEmpty() }
    suspend fun setHfToken(token: String) { store.edit { it[hfTokenKey] = token.trim() } }

    /** Context window of the chat model (tokens): longer chats need more memory. */
    val chatContextTokens: Flow<Int> = store.data.map { it[chatContextKey] ?: DEFAULT_CHAT_CONTEXT }
    suspend fun setChatContextTokens(n: Int) { store.edit { it[chatContextKey] = n } }

    /** Run models on the NPU when their NPU build for this phone is downloaded (else GPU, then CPU). */
    val useNpu: Flow<Boolean> = store.data.map { it[useNpuKey] ?: true }
    suspend fun setUseNpu(on: Boolean) { store.edit { it[useNpuKey] = on } }

    /** Packages that have used the host, with the categories each used (for warnings and notifications). */
    val clients: Flow<Map<String, Set<String>>> = store.data.map { p ->
        p[clientsKey].orEmpty().associateWith { p[clientCategoriesKey(it)].orEmpty() }
    }

    suspend fun recordClientUse(pkg: String, category: String?) {
        store.edit { p ->
            p[clientsKey] = p[clientsKey].orEmpty() + pkg
            if (category != null) p[clientCategoriesKey(pkg)] = p[clientCategoriesKey(pkg)].orEmpty() + category
        }
    }

    suspend fun forgetClient(pkg: String) {
        store.edit { p ->
            p[clientsKey] = p[clientsKey].orEmpty() - pkg
            p.remove(clientCategoriesKey(pkg))
        }
    }

    companion object {
        const val DEFAULT_CHAT_CONTEXT = 4096
        val CHAT_CONTEXT_CHOICES = listOf(2048, 4096, 8192)
    }
}
