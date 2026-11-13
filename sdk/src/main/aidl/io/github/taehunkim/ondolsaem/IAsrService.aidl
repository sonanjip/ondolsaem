package io.github.taehunkim.ondolsaem;

import io.github.taehunkim.ondolsaem.ICueCallback;

/**
 * Category "asr". Audio is raw PCM: 16 kHz, mono, 16-bit little endian.
 * Voice activity detection works without a downloaded model; transcription needs one (canTranscribe()).
 */
interface IAsrService {
    boolean canTranscribe() = 1;
    /** Speech time ranges as [start0, end0, start1, end1, ...] in milliseconds. */
    long[] speechSpans(in ParcelFileDescriptor pcm, int priority) = 2;
    /** Blocks until done; utterances from fromMs on are passed to the callback. */
    void transcribe(in ParcelFileDescriptor pcm, long fromMs, ICueCallback callback, int priority) = 3;
    /** API 5. Request-scoped language: auto, ko, en, ja, zh, yue. Legacy transcribe always uses auto. */
    void transcribeWithLanguage(in ParcelFileDescriptor pcm, long fromMs, ICueCallback callback, int priority, String language) = 4;
}
