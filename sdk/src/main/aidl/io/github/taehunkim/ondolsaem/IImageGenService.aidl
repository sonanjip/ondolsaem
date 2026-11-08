package io.github.taehunkim.ondolsaem;

import android.os.ParcelFileDescriptor;
import io.github.taehunkim.ondolsaem.ModelInfo;

/** Category "image_gen" (API 4). Text-to-image generation. */
interface IImageGenService {
    ModelInfo getActiveModel() = 1;
    /** Generates an image for [prompt] with [seed], returned as a PNG stream. */
    ParcelFileDescriptor generateImage(String prompt, long seed, int priority) = 2;
}
