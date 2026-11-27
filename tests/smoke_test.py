"""Smoke test: load an nnet3 model and stream-decode WAV files like pynab does.

Usage: python3 tests/smoke_test.py MODEL_DIR WAV [WAV ...]

pynab (nabd/asr.py) feeds 16 kHz int16 chunks as float32 numpy arrays while
recording, then finalizes. This mirrors that path, plus an empty final chunk
(allowed since 0.6.0, which uses typed memoryviews).
"""

import array
import sys
import time
import wave

import numpy as np

from kaldiasr.nnet3 import KaldiNNet3OnlineDecoder, KaldiNNet3OnlineModel

CHUNK = 1600  # 100 ms at 16 kHz


def decode_streaming(decoder, path):
    with wave.open(path, "rb") as wavf:
        assert wavf.getnchannels() == 1 and wavf.getsampwidth() == 2
        rate = wavf.getframerate()
        pcm = np.frombuffer(wavf.readframes(wavf.getnframes()), dtype="<i2")
    samples = pcm.astype(np.float32)
    for start in range(0, len(samples), CHUNK):
        decoder.decode(rate, samples[start : start + CHUNK], False)
    decoder.decode(rate, np.zeros(0, dtype=np.float32), True)
    return decoder.get_decoded_string()


def main(model_dir, wavs):
    t0 = time.time()
    model = KaldiNNet3OnlineModel(model_dir, max_mem=20000)
    print("model loaded in %.1fs" % (time.time() - t0))

    for path in wavs:
        decoder = KaldiNNet3OnlineDecoder(model)
        t0 = time.time()
        text, likelihood = decode_streaming(decoder, path)
        assert isinstance(text, str)
        print("%s -> %r (likelihood %.3f, %.1fs)" % (path, text, likelihood, time.time() - t0))

    # Non-numpy buffers are accepted too (no NumPy C-API dependency).
    decoder = KaldiNNet3OnlineDecoder(model)
    decoder.decode(16000, array.array("f", [0.0] * CHUNK), True)
    print("array.array input OK")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
