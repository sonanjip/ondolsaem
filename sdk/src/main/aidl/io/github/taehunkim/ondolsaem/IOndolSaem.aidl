package io.github.taehunkim.ondolsaem;

import io.github.taehunkim.ondolsaem.ClientInfo;
import io.github.taehunkim.ondolsaem.HostInfo;
import io.github.taehunkim.ondolsaem.IClientCallback;
import io.github.taehunkim.ondolsaem.ILease;
import io.github.taehunkim.ondolsaem.ModelInfo;

/**
 * Entry point of the OndolSaem model host (bound service, action OndolSaem.ACTION_BIND).
 *
 * Versioning: methods are only ever appended (explicit transaction ids), never reordered, removed or changed; each
 * method notes the API level that added it. A client first calls hello() (API 1), which exchanges versions: the
 * client must not call methods above HostInfo.apiVersion, and the host refuses clients below HostInfo.minClientApi.
 * apiVersion() stays the cheapest check and never changes.
 * Errors arrive as IllegalStateException encoded by HostErrors (codes in OndolSaem.Errors), or SecurityException.
 */
interface IOndolSaem {
    int apiVersion() = 1;
    /** The active model of every category the host knows. */
    ModelInfo[] categories() = 2;
    /** Null for an unknown category. */
    @nullable ModelInfo activeModel(String category) = 3;
    /** IEmbeddingService, IChatService, IOcrService, IAsrService, (API 2) IFaceService, (API 3) IDecisionService or (API 4) IImageGenService; null for an unknown category. */
    @nullable IBinder getCapability(String category) = 4;
    /**
     * API 1. Exchanges versions and registers the callback for model changes. Throws CLIENT_TOO_OLD when the
     * client's API is below the host's minimum.
     */
    HostInfo hello(in ClientInfo client, IClientCallback callback) = 5;
    void unregisterClient(IClientCallback callback) = 6;
    /**
     * Loads the category's model now and keeps it loaded until the lease is released or [clientToken] dies
     * (any Binder object of the client; the host watches it so a crashed client does not pin the model).
     */
    ILease acquire(String category, IBinder clientToken) = 7;
}
