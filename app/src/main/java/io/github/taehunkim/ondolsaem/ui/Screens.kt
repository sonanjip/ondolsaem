package io.github.taehunkim.ondolsaem.ui

import android.text.format.DateUtils
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import io.github.taehunkim.ondolsaem.BuildConfig
import io.github.taehunkim.ondolsaem.R
import io.github.taehunkim.ondolsaem.catalog.ChangeImpact
import io.github.taehunkim.ondolsaem.catalog.ModelCatalog
import io.github.taehunkim.ondolsaem.catalog.ModelCategories
import io.github.taehunkim.ondolsaem.catalog.ModelCategory
import io.github.taehunkim.ondolsaem.catalog.ModelSource
import io.github.taehunkim.ondolsaem.catalog.ModelSpec
import io.github.taehunkim.ondolsaem.download.DownloadException
import io.github.taehunkim.ondolsaem.download.ModelState
import io.github.taehunkim.ondolsaem.runtime.Residency
import io.github.taehunkim.ondolsaem.settings.HostSettings
import kotlinx.coroutines.delay

fun formatBytes(b: Long): String = when {
    b >= 1L shl 30 -> "%.1f GB".format(b / (1L shl 30).toDouble())
    b >= 1L shl 20 -> "%.0f MB".format(b / (1L shl 20).toDouble())
    else -> "%.0f KB".format(b / 1024.0)
}

fun formatKb(kb: Long) = formatBytes(kb * 1024)

@Composable
private fun SectionCard(title: String, content: @Composable () -> Unit) {
    Card(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 6.dp)) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(title, style = MaterialTheme.typography.titleMedium)
            content()
        }
    }
}

@Composable
private fun KeyValue(key: String, value: String) {
    Row(Modifier.fillMaxWidth()) {
        Text(key, Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

/** Memory-first overview: device memory, the host process, and every loaded model with the apps holding it. */
@Composable
fun DashboardScreen(vm: HostViewModel) {
    val mem by vm.memory.collectAsState()
    val resident by vm.resident.collectAsState()
    val active by vm.activeIds.collectAsState()
    val residency by vm.residency.collectAsState()
    var now by remember { mutableLongStateOf(System.currentTimeMillis()) }
    LaunchedEffect(Unit) { while (true) { delay(1_000); now = System.currentTimeMillis() } }

    val keepBlocked by vm.keepAliveBlocked.collectAsState()
    val ctx = androidx.compose.ui.platform.LocalContext.current

    LazyColumn(Modifier.testTag("dashboard")) {
        if (keepBlocked) item {
            SectionCard(stringResource(R.string.notif_keep_blocked_title)) {
                Text(stringResource(R.string.dash_keep_blocked), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (vm.batteryOptimized()) OutlinedButton(onClick = {
                        @android.annotation.SuppressLint("BatteryLife")
                        val i = android.content.Intent(android.provider.Settings.ACTION_REQUEST_IGNORE_BATTERY_OPTIMIZATIONS, android.net.Uri.parse("package:" + ctx.packageName))
                        runCatching { ctx.startActivity(i) }
                    }) { Text(stringResource(R.string.dash_keep_battery)) }
                    OutlinedButton(onClick = vm::retryKeepAlive) { Text(stringResource(R.string.dash_keep_retry)) }
                }
            }
        }
        item {
            SectionCard(stringResource(R.string.dash_device)) {
                val used = (mem.deviceTotal - mem.deviceAvail).coerceAtLeast(0)
                LinearProgressIndicator(
                    progress = { if (mem.deviceTotal > 0) used.toFloat() / mem.deviceTotal else 0f },
                    modifier = Modifier.fillMaxWidth(),
                )
                KeyValue(stringResource(R.string.dash_available), "${formatBytes(mem.deviceAvail)} / ${formatBytes(mem.deviceTotal)}")
                KeyValue(stringResource(R.string.dash_low_threshold), formatBytes(mem.deviceThreshold))
                if (mem.lowMemory) Text(stringResource(R.string.dash_low_memory), color = MaterialTheme.colorScheme.error)
            }
        }
        item {
            SectionCard(stringResource(R.string.dash_process)) {
                KeyValue(stringResource(R.string.dash_total), formatKb(mem.processPssKb))
                KeyValue(stringResource(R.string.dash_native), formatKb(mem.nativeKb))
                KeyValue(stringResource(R.string.dash_graphics), formatKb(mem.graphicsKb))
                KeyValue(stringResource(R.string.dash_java), formatKb(mem.javaKb))
            }
        }
        items(ModelCategories.all, key = { it.id }) { cat ->
            val st = resident[cat.id]
            val spec = vm.spec(active[cat.id])
            SectionCard(stringResource(cat.titleRes)) {
                Text(spec?.displayName ?: stringResource(R.string.dash_no_model), style = MaterialTheme.typography.bodyMedium)
                if (st == null) {
                    Text(stringResource(R.string.dash_not_loaded), style = MaterialTheme.typography.bodySmall)
                } else {
                    KeyValue(stringResource(R.string.dash_backend), st.backend ?: "-")
                    KeyValue(stringResource(R.string.dash_model_memory), st.loadedKb?.let { "~" + formatKb(it) } ?: "-")
                    val holders = st.holders.keys.map { if (it == "host") stringResource(R.string.app_name) else vm.label(it) }
                    KeyValue(stringResource(R.string.dash_in_use_by), holders.joinToString().ifEmpty { stringResource(R.string.dash_idle) })
                    st.unloadAt?.let { at ->
                        KeyValue(stringResource(R.string.dash_unloads_in), DateUtils.formatElapsedTime(((at - now) / 1000).coerceAtLeast(0)))
                    }
                }
                KeyValue(stringResource(R.string.dash_policy), stringResource((residency[cat.id] ?: cat.defaultResidency).labelRes))
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (st == null) OutlinedButton(onClick = { vm.load(cat.id) }, enabled = spec != null) { Text(stringResource(R.string.action_load)) }
                    else OutlinedButton(onClick = { vm.unload(cat.id) }, enabled = st.holders.isEmpty()) { Text(stringResource(R.string.action_unload)) }
                }
            }
        }
    }
}

/** Category cards: description, residency, curated models (download, delete, choose the default). */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ModelsScreen(vm: HostViewModel) {
    val states by vm.modelStates.collectAsState()
    val active by vm.activeIds.collectAsState()
    val residency by vm.residency.collectAsState()
    val ctx by vm.chatContext.collectAsState()
    val clients by vm.clients.collectAsState() // for the model-change warning
    val npuStates by vm.npuStates.collectAsState()
    val useNpu by vm.useNpu.collectAsState()
    var meteredFor by remember { mutableStateOf<ModelSpec?>(null) }
    var meteredNpu by remember { mutableStateOf<ModelSpec?>(null) }
    var switchTo by remember { mutableStateOf<Pair<ModelCategory, ModelSpec>?>(null) }

    LazyColumn(Modifier.testTag("models")) {
        items(ModelCategories.all, key = { it.id }) { cat ->
            SectionCard(stringResource(cat.titleRes)) {
                Text(stringResource(cat.descriptionRes), style = MaterialTheme.typography.bodySmall)
                Text(stringResource(R.string.models_residency), style = MaterialTheme.typography.labelLarge)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Residency.entries.forEach { r ->
                        FilterChip(
                            selected = (residency[cat.id] ?: cat.defaultResidency) == r,
                            onClick = { vm.setResidency(cat.id, r) },
                            label = { Text(stringResource(r.labelRes)) },
                        )
                    }
                }
                if (cat.id == io.github.taehunkim.ondolsaem.sdk.OndolSaem.Categories.LLM) {
                    Text(stringResource(R.string.models_chat_context), style = MaterialTheme.typography.labelLarge)
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        HostSettings.CHAT_CONTEXT_CHOICES.forEach { n ->
                            FilterChip(selected = ctx == n, onClick = { vm.setChatContext(n) }, label = { Text("$n") })
                        }
                    }
                }
                ModelCatalog.ofCategory(cat.id).forEach { spec ->
                    ModelRow(
                        spec, states[spec.id] ?: ModelState.NotInstalled, active[cat.id] == spec.id,
                        onSelect = {
                            if (active[cat.id] != spec.id) {
                                if (cat.changeImpact == ChangeImpact.NONE) vm.select(cat.id, spec.id) else switchTo = cat to spec
                            }
                        },
                        onDownload = { if (vm.isMetered()) meteredFor = spec else vm.download(spec.id, wifiOnly = false) },
                        onCancel = { vm.cancelDownload(spec.id) },
                        onDelete = { vm.delete(spec.id) },
                    )
                    val npuState = npuStates[spec.id]
                    val soc = vm.npuDevice?.soc
                    if (npuState != null && soc != null && useNpu && states[spec.id] is ModelState.Installed) NpuRow(
                        soc, vm.npuSize(spec.id), npuState, useNpu,
                        onDownload = { if (vm.isMetered()) meteredNpu = spec else vm.downloadNpu(spec.id, wifiOnly = false) },
                        onCancel = { vm.cancelNpu(spec.id) },
                        onDelete = { vm.deleteNpu(spec.id) },
                    )
                }
            }
        }
    }

    meteredFor?.let { spec ->
        MeteredDialog(
            spec.displayName,
            spec.sizeBytes?.let(::formatBytes) ?: stringResource(R.string.model_aicore_managed),
            onDownload = { wifiOnly -> vm.download(spec.id, wifiOnly); meteredFor = null },
            onCancel = { meteredFor = null },
        )
    }

    meteredNpu?.let { spec ->
        MeteredDialog(
            stringResource(R.string.npu_build_name, spec.displayName), formatBytes(vm.npuSize(spec.id)),
            onDownload = { wifiOnly -> vm.downloadNpu(spec.id, wifiOnly); meteredNpu = null },
            onCancel = { meteredNpu = null },
        )
    }

    switchTo?.let { (cat, spec) ->
        val apps = clients.filter { cat.id in it.value }.keys.map(vm::label).sorted()
        val appsText = if (apps.isEmpty()) stringResource(R.string.switch_no_apps) else apps.joinToString()
        AlertDialog(
            onDismissRequest = { switchTo = null },
            title = { Text(stringResource(R.string.switch_title, stringResource(cat.titleRes))) },
            text = {
                Text(
                    stringResource(
                        if (cat.changeImpact == ChangeImpact.REINDEX_REQUIRED) R.string.switch_body_reindex else R.string.switch_body_derived,
                        spec.displayName, appsText,
                    ),
                )
            },
            confirmButton = { TextButton(onClick = { vm.select(cat.id, spec.id); switchTo = null }) { Text(stringResource(R.string.switch_confirm)) } },
            dismissButton = { TextButton(onClick = { switchTo = null }) { Text(stringResource(R.string.action_cancel)) } },
        )
    }
}

@Composable
private fun ModelRow(
    spec: ModelSpec, state: ModelState, selected: Boolean,
    onSelect: () -> Unit, onDownload: () -> Unit, onCancel: () -> Unit, onDelete: () -> Unit,
) {
    val usable = state is ModelState.Installed
    val isAiCore = spec.source is ModelSource.AiCore
    Column(Modifier.fillMaxWidth().padding(top = 6.dp).testTag("model_${spec.id}")) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            RadioButton(
                selected = selected,
                onClick = onSelect,
                enabled = spec.source !is ModelSource.Planned && !(isAiCore && state is ModelState.Unavailable),
            )
            Column(Modifier.weight(1f)) {
                Text(spec.displayName, style = MaterialTheme.typography.bodyLarge)
                val sizeLabel = spec.sizeBytes?.let(::formatBytes) ?: if (isAiCore) stringResource(R.string.model_aicore_managed) else null
                Text(
                    listOfNotNull(sizeLabel, spec.license).joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }
        Text(stringResource(spec.descriptionRes), style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(start = 48.dp))
        Row(Modifier.padding(start = 48.dp), verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            when (state) {
                ModelState.Unavailable -> Text(
                    stringResource(if (isAiCore) R.string.model_aicore_unavailable else R.string.model_coming_soon),
                    style = MaterialTheme.typography.labelMedium,
                )
                ModelState.Installed -> {
                    when (spec.source) {
                        is ModelSource.Bundled -> Text(stringResource(R.string.model_bundled), style = MaterialTheme.typography.labelMedium)
                        is ModelSource.AiCore -> Text(stringResource(R.string.model_aicore_ready), style = MaterialTheme.typography.labelMedium)
                        else -> {
                            Text(stringResource(R.string.model_installed), style = MaterialTheme.typography.labelMedium)
                            TextButton(onClick = onDelete) { Text(stringResource(R.string.model_delete)) }
                        }
                    }
                }
                ModelState.NotInstalled -> OutlinedButton(onClick = onDownload) {
                    Text(stringResource(if (isAiCore) R.string.model_aicore_download else R.string.model_download))
                }
                ModelState.Verifying -> {
                    LinearProgressIndicator(Modifier.weight(1f))
                    Text(stringResource(R.string.model_verifying), style = MaterialTheme.typography.labelMedium)
                }
                is ModelState.Converting -> {
                    LinearProgressIndicator({ if (state.total > 0) state.done.toFloat() / state.total else 0f }, Modifier.weight(1f))
                    Text(stringResource(R.string.model_converting), style = MaterialTheme.typography.labelMedium)
                }
                ModelState.WaitingForWifi -> {
                    Text(stringResource(R.string.model_waiting_wifi), style = MaterialTheme.typography.labelMedium)
                    TextButton(onClick = onCancel) { Text(stringResource(R.string.action_cancel)) }
                }
                is ModelState.Downloading -> {
                    val pct = if (state.total > 0) (state.downloaded * 100 / state.total).toInt() else 0
                    Column(Modifier.weight(1f)) {
                        if (state.total > 0) {
                            LinearProgressIndicator(progress = { pct / 100f }, modifier = Modifier.fillMaxWidth())
                            Text(stringResource(R.string.model_downloading, pct), style = MaterialTheme.typography.labelMedium)
                        } else {
                            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                            Text(stringResource(R.string.model_downloading, 0), style = MaterialTheme.typography.labelMedium)
                        }
                    }
                    TextButton(onClick = onCancel) { Text(stringResource(R.string.action_cancel)) }
                }
                is ModelState.Failed -> {
                    Text(
                        stringResource(
                            when (state.kind) {
                                DownloadException.Kind.AUTH -> R.string.model_err_auth
                                DownloadException.Kind.NOT_FOUND -> R.string.model_err_not_found
                                DownloadException.Kind.NETWORK -> R.string.model_err_network
                                DownloadException.Kind.CORRUPT -> R.string.model_err_corrupt
                                else -> R.string.model_err_http
                            },
                        ),
                        color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f),
                    )
                    OutlinedButton(onClick = onDownload) { Text(stringResource(R.string.model_retry)) }
                }
            }
        }
        if (selected && !usable && state !is ModelState.Unavailable) {
            Text(stringResource(R.string.model_selected_missing), color = MaterialTheme.colorScheme.error,
                style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(start = 48.dp))
        }
    }
}

/** A model's NPU build for this phone ([soc]): download, progress, delete. Shown under an installed model. */
@Composable
private fun NpuRow(soc: String, size: Long, state: ModelState, useNpu: Boolean, onDownload: () -> Unit, onCancel: () -> Unit, onDelete: () -> Unit) {
    Column(Modifier.fillMaxWidth().padding(start = 48.dp).testTag("npu_row")) {
        if (state == ModelState.Unavailable) {
            Text(stringResource(R.string.npu_none_for_chip, soc), style = MaterialTheme.typography.bodySmall)
            return@Column
        }
        Text(stringResource(R.string.npu_build, soc), style = MaterialTheme.typography.labelLarge)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            when (state) {
                ModelState.Installed -> {
                    Text(stringResource(if (useNpu) R.string.npu_installed else R.string.npu_installed_off), style = MaterialTheme.typography.labelMedium, modifier = Modifier.weight(1f))
                    TextButton(onClick = onDelete) { Text(stringResource(R.string.model_delete)) }
                }
                is ModelState.Downloading -> {
                    val pct = if (state.total > 0) (state.downloaded * 100 / state.total).toInt() else 0
                    Column(Modifier.weight(1f)) {
                        LinearProgressIndicator(progress = { pct / 100f }, modifier = Modifier.fillMaxWidth())
                        Text(stringResource(R.string.model_downloading, pct), style = MaterialTheme.typography.labelMedium)
                    }
                    TextButton(onClick = onCancel) { Text(stringResource(R.string.action_cancel)) }
                }
                ModelState.Verifying -> {
                    LinearProgressIndicator(Modifier.weight(1f))
                    Text(stringResource(R.string.model_verifying), style = MaterialTheme.typography.labelMedium)
                }
                ModelState.WaitingForWifi -> {
                    Text(stringResource(R.string.model_waiting_wifi), style = MaterialTheme.typography.labelMedium)
                    TextButton(onClick = onCancel) { Text(stringResource(R.string.action_cancel)) }
                }
                is ModelState.Failed -> {
                    Text(stringResource(R.string.npu_failed), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    OutlinedButton(onClick = onDownload) { Text(stringResource(R.string.model_retry)) }
                }
                else -> OutlinedButton(onClick = onDownload) { Text(stringResource(R.string.npu_download, formatBytes(size))) }
            }
        }
    }
}

/** Apps that used the host, with their categories and request statistics. */
@Composable
fun AppsScreen(vm: HostViewModel) {
    val clients by vm.clients.collectAsState()
    val usage by vm.registry.usage.collectAsState()
    val versions by vm.registry.versions.collectAsState()
    val approvals by vm.approvals.collectAsState()
    LazyColumn(Modifier.testTag("apps")) {
        if (approvals.isNotEmpty()) item {
            Text(stringResource(R.string.approvals_title), Modifier.padding(start = 16.dp, top = 12.dp, end = 16.dp), style = MaterialTheme.typography.titleSmall)
        }
        // waiting requests first, then the newest
        items(approvals.values.sortedWith(compareBy<io.github.taehunkim.ondolsaem.service.Approval> { it.state != io.github.taehunkim.ondolsaem.service.Approval.State.PENDING }.thenByDescending { it.requestedAt }), key = { "approval_" + it.pkg }) { a ->
            ApprovalCard(vm, a)
        }
        if (clients.isEmpty() && approvals.isEmpty()) item {
            Text(stringResource(R.string.apps_empty), Modifier.padding(16.dp))
        }
        items(clients.keys.sorted()) { pkg ->
            SectionCard(vm.label(pkg)) {
                Text(pkg, style = MaterialTheme.typography.bodySmall)
                versions[pkg]?.let { v ->
                    Text(
                        stringResource(R.string.apps_sdk, v.sdkVersion, v.apiVersion),
                        style = MaterialTheme.typography.bodySmall,
                        color = if (v.apiVersion < io.github.taehunkim.ondolsaem.HostContainer.MIN_CLIENT_API) MaterialTheme.colorScheme.error
                        else MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                val cats = clients[pkg].orEmpty()
                cats.sorted().forEach { catId ->
                    val cat = ModelCategories.byId(catId)
                    val u = usage[pkg]?.get(catId)
                    val name = cat?.let { stringResource(it.titleRes) } ?: catId
                    KeyValue(
                        name,
                        if (u == null) stringResource(R.string.apps_no_requests)
                        else stringResource(R.string.apps_usage, u.requests, u.averageMs,
                            DateUtils.getRelativeTimeSpanString(u.lastUse).toString()),
                    )
                }
                TextButton(onClick = { vm.forgetClient(pkg) }) { Text(stringResource(R.string.apps_forget)) }
            }
        }
        item { Text(stringResource(R.string.apps_note), Modifier.padding(16.dp), style = MaterialTheme.typography.bodySmall) }
    }
}

@Composable
private fun ApprovalCard(vm: HostViewModel, a: io.github.taehunkim.ondolsaem.service.Approval) {
    SectionCard(vm.label(a.pkg)) {
        Text(a.pkg, style = MaterialTheme.typography.bodySmall)
        KeyValue(stringResource(R.string.approvals_cert), a.certSha256.take(23).chunked(2).joinToString(":") + "…")
        KeyValue(stringResource(R.string.approvals_requested), DateUtils.getRelativeTimeSpanString(a.requestedAt).toString())
        Text(
            stringResource(
                when (a.state) {
                    io.github.taehunkim.ondolsaem.service.Approval.State.PENDING -> R.string.approvals_pending
                    io.github.taehunkim.ondolsaem.service.Approval.State.ALLOWED -> R.string.approvals_allowed
                    io.github.taehunkim.ondolsaem.service.Approval.State.DENIED -> R.string.approvals_denied
                },
            ),
            style = MaterialTheme.typography.bodyMedium,
            color = if (a.state == io.github.taehunkim.ondolsaem.service.Approval.State.PENDING) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (a.state != io.github.taehunkim.ondolsaem.service.Approval.State.ALLOWED) androidx.compose.material3.Button(onClick = { vm.decideApproval(a.pkg, true) }, Modifier.testTag("allow_${a.pkg}")) {
                Text(stringResource(R.string.approvals_allow))
            }
            if (a.state != io.github.taehunkim.ondolsaem.service.Approval.State.DENIED) OutlinedButton(onClick = { vm.decideApproval(a.pkg, false) }, Modifier.testTag("deny_${a.pkg}")) {
                Text(stringResource(if (a.state == io.github.taehunkim.ondolsaem.service.Approval.State.ALLOWED) R.string.approvals_revoke else R.string.approvals_deny))
            }
            TextButton(onClick = { vm.forgetApproval(a.pkg) }) { Text(stringResource(R.string.apps_forget)) }
        }
    }
}

@Composable
fun SettingsScreen(vm: HostViewModel) {
    val token by vm.hfToken.collectAsState()
    var edit by rememberSaveable(token) { mutableStateOf(token) }
    val context = LocalContext.current
    LazyColumn(Modifier.testTag("settings")) {
        item {
            SectionCard(stringResource(R.string.settings_hf_token)) {
                Text(stringResource(R.string.settings_hf_token_desc), style = MaterialTheme.typography.bodySmall)
                OutlinedTextField(
                    value = edit, onValueChange = { edit = it }, singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    modifier = Modifier.fillMaxWidth(),
                )
                Row {
                    Spacer(Modifier.weight(1f))
                    TextButton(onClick = { vm.setHfToken(edit) }, enabled = edit != token) { Text(stringResource(R.string.action_save)) }
                }
            }
        }
        item { NpuSettings(vm) }
        item {
            SectionCard(stringResource(R.string.settings_about)) {
                KeyValue(stringResource(R.string.settings_version), BuildConfig.VERSION_NAME)
                KeyValue(stringResource(R.string.settings_api), io.github.taehunkim.ondolsaem.sdk.OndolSaem.API_VERSION.toString())
                KeyValue(stringResource(R.string.settings_min_client_api), io.github.taehunkim.ondolsaem.HostContainer.MIN_CLIENT_API.toString())
                Text(stringResource(R.string.settings_about_body), style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.width(1.dp))
                Text(context.packageName, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun NpuSettings(vm: HostViewModel) {
    val useNpu by vm.useNpu.collectAsState()
    val npuStates by vm.npuStates.collectAsState()
    val device = vm.npuDevice
    SectionCard(stringResource(R.string.npu_title)) {
        if (device == null) {
            Text(stringResource(R.string.npu_unsupported), style = MaterialTheme.typography.bodySmall)
            NpuSelfCheckSection(vm)
            return@SectionCard
        }
        KeyValue(stringResource(R.string.npu_chip), device.soc)
        if (!device.usable) {
            Text(stringResource(R.string.npu_unknown_qualcomm), style = MaterialTheme.typography.bodySmall)
            return@SectionCard
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(stringResource(R.string.npu_use), Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
            androidx.compose.material3.Switch(checked = useNpu, onCheckedChange = vm::setUseNpu, modifier = Modifier.testTag("npu_switch"))
        }
        Text(stringResource(R.string.npu_use_desc), style = MaterialTheme.typography.bodySmall)
        val check by vm.npuCheck.collectAsState()
        val found = check.result?.builds?.keys.orEmpty().mapNotNull { vm.spec(it)?.displayName }
        KeyValue(
            stringResource(R.string.npu_builds),
            when {
                check.running -> stringResource(R.string.npu_checking)
                check.result == null -> stringResource(R.string.npu_not_checked)
                else -> DateUtils.getRelativeTimeSpanString(check.result!!.checkedAt).toString()
            },
        )
        if (check.result != null) Text(
            if (found.isEmpty()) stringResource(R.string.npu_found_none, device.soc) else stringResource(R.string.npu_found, device.soc, found.joinToString()),
            style = MaterialTheme.typography.bodySmall,
        )
        if (check.failed != null) Text(stringResource(R.string.npu_check_failed), color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = vm::recheckNpu, enabled = useNpu && !check.running, modifier = Modifier.testTag("npu_recheck")) {
            Text(stringResource(R.string.npu_recheck))
        }
        if (device.vendor == io.github.taehunkim.ondolsaem.engine.NpuVendor.QUALCOMM) {
            val bytes = remember(npuStates) { vm.npuRuntimeBytes() }
            KeyValue(
                stringResource(R.string.npu_qnn),
                if (bytes > 0) "${io.github.taehunkim.ondolsaem.download.NpuRuntime.QNN_VERSION} · ${formatBytes(bytes)}" else stringResource(R.string.npu_qnn_missing),
            )
            Text(stringResource(R.string.npu_qnn_desc, device.hexagon ?: "?"), style = MaterialTheme.typography.bodySmall)
        }
        NpuSelfCheckSection(vm)
    }
}

/** "Check the NPU": runs [io.github.taehunkim.ondolsaem.runtime.NpuSelfCheck] and shows the report to copy or share. */
@Composable
private fun NpuSelfCheckSection(vm: HostViewModel) {
    val report by vm.npuReport.collectAsState()
    val context = LocalContext.current
    Text(stringResource(R.string.npu_selfcheck_desc), style = MaterialTheme.typography.bodySmall)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        OutlinedButton(onClick = vm::runNpuSelfCheck, enabled = report?.running != true, modifier = Modifier.testTag("npu_selfcheck")) {
            Text(stringResource(R.string.npu_selfcheck))
        }
        val r = report
        if (r != null && !r.running) {
            TextButton(onClick = {
                val cm = context.getSystemService(android.content.ClipboardManager::class.java)
                cm.setPrimaryClip(android.content.ClipData.newPlainText("OndolSaem NPU check", r.text))
            }) { Text(stringResource(R.string.npu_selfcheck_copy)) }
            TextButton(onClick = {
                val send = android.content.Intent(android.content.Intent.ACTION_SEND).setType("text/plain")
                    .putExtra(android.content.Intent.EXTRA_SUBJECT, "OndolSaem NPU check")
                    .putExtra(android.content.Intent.EXTRA_TEXT, r.text)
                runCatching { context.startActivity(android.content.Intent.createChooser(send, null)) }
            }) { Text(stringResource(R.string.npu_selfcheck_share)) }
        }
    }
    report?.let { r ->
        if (r.running) {
            LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(stringResource(R.string.npu_selfcheck_running), style = MaterialTheme.typography.labelMedium)
        }
        androidx.compose.foundation.text.selection.SelectionContainer {
            Text(
                r.text, style = MaterialTheme.typography.bodySmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace),
                modifier = Modifier.testTag("npu_report"),
            )
        }
    }
}

/** Asks before downloading [what] ([size]) on a metered network: now, or wait for Wi-Fi. */
@Composable
fun MeteredDialog(what: String, size: String, onDownload: (wifiOnly: Boolean) -> Unit, onCancel: () -> Unit) {
    AlertDialog(
        onDismissRequest = onCancel,
        title = { Text(stringResource(R.string.metered_title)) },
        text = { Text(stringResource(R.string.metered_body, what, size)) },
        confirmButton = { TextButton(onClick = { onDownload(false) }) { Text(stringResource(R.string.metered_now)) } },
        dismissButton = {
            Row {
                TextButton(onClick = onCancel) { Text(stringResource(R.string.action_cancel)) }
                TextButton(onClick = { onDownload(true) }) { Text(stringResource(R.string.metered_wifi)) }
            }
        },
    )
}
