/**
 * Microphone capture that produces the format the API actually accepts:
 * 16 kHz mono 16-bit PCM in a WAV container.
 *
 * MediaRecorder cannot emit WAV - it gives WebM/Opus at the hardware sample rate
 * (usually 44.1 or 48 kHz). Rather than reach for an AudioWorklet to intercept raw
 * frames, this records normally and converts afterwards: decode the blob, resample
 * through OfflineAudioContext, then write the WAV header ourselves. Conversion of a
 * few seconds of speech takes a handful of milliseconds and needs no worklet file
 * served alongside the app, which is one less thing to break in deployment.
 */

const TARGET_SAMPLE_RATE = 16_000;

export class RecorderError extends Error {}

/** MIME type the browser will actually give us, preferring Opus. */
function pickMimeType(): string | undefined {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  return candidates.find((type) => MediaRecorder.isTypeSupported(type));
}

export class MicRecorder {
  private stream: MediaStream | null = null;
  private recorder: MediaRecorder | null = null;
  private chunks: Blob[] = [];

  get active(): boolean {
    return this.recorder?.state === "recording";
  }

  /** The live stream, so a level meter can read it without opening the mic twice. */
  get mediaStream(): MediaStream | null {
    return this.stream;
  }

  async start(): Promise<void> {
    if (this.active) return;

    if (!navigator.mediaDevices?.getUserMedia) {
      throw new RecorderError(
        "This browser cannot capture audio. Try Chrome, Edge or Safari over HTTPS."
      );
    }

    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (error) {
      const name = (error as DOMException)?.name;
      if (name === "NotAllowedError" || name === "SecurityError") {
        throw new RecorderError(
          "Microphone permission was denied. Allow it in the browser's site settings and try again."
        );
      }
      if (name === "NotFoundError") {
        throw new RecorderError("No microphone was found.");
      }
      throw new RecorderError(`Could not open the microphone: ${name ?? error}`);
    }

    const mimeType = pickMimeType();
    this.chunks = [];
    this.recorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : undefined);
    this.recorder.ondataavailable = (event) => {
      if (event.data.size > 0) this.chunks.push(event.data);
    };
    // Timeslice so a long recording still flushes data periodically rather than
    // holding everything until stop().
    this.recorder.start(250);
  }

  /** Stop, and return the recording as a 16 kHz mono WAV. */
  async stop(): Promise<Blob> {
    const recorder = this.recorder;
    if (!recorder) throw new RecorderError("Recording was never started.");

    const recorded = await new Promise<Blob>((resolve) => {
      recorder.onstop = () => resolve(new Blob(this.chunks, { type: recorder.mimeType }));
      recorder.stop();
    });

    this.release();

    if (recorded.size === 0) {
      throw new RecorderError("Nothing was recorded. Check the microphone and try again.");
    }
    return encodeWav(trimSilence(await resampleToMono16k(recorded)));
  }

  /** Drop the mic without producing audio, e.g. when the user cancels. */
  cancel(): void {
    if (this.recorder?.state === "recording") this.recorder.stop();
    this.release();
    this.chunks = [];
  }

  private release(): void {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    this.recorder = null;
  }
}

/** Decode whatever the browser recorded and resample it to 16 kHz mono. */
async function resampleToMono16k(blob: Blob): Promise<Float32Array> {
  const bytes = await blob.arrayBuffer();

  const decodeContext = new AudioContext();
  let decoded: AudioBuffer;
  try {
    decoded = await decodeContext.decodeAudioData(bytes);
  } catch {
    throw new RecorderError("The recorded audio could not be decoded.");
  } finally {
    void decodeContext.close();
  }

  const frames = Math.ceil(decoded.duration * TARGET_SAMPLE_RATE);
  if (frames === 0) {
    throw new RecorderError("The recording was empty.");
  }

  // OfflineAudioContext does the resampling and the channel mixdown in one pass.
  const offline = new OfflineAudioContext(1, frames, TARGET_SAMPLE_RATE);
  const source = offline.createBufferSource();
  source.buffer = decoded;
  source.connect(offline.destination);
  source.start();

  const rendered = await offline.startRendering();
  return rendered.getChannelData(0);
}

/** Drop leading/trailing hush so the 2 s send-pause is not sent to STT. */
function trimSilence(samples: Float32Array, threshold = 0.02): Float32Array {
  let start = 0;
  let end = samples.length - 1;
  while (start < end && Math.abs(samples[start]) < threshold) start += 1;
  while (end > start && Math.abs(samples[end]) < threshold) end -= 1;
  const pad = 2_400; // 150 ms at 16 kHz
  start = Math.max(0, start - pad);
  end = Math.min(samples.length - 1, end + pad);
  if (end - start < TARGET_SAMPLE_RATE * 0.35) {
    return samples;
  }
  return samples.subarray(start, end + 1);
}

/** Write a 16-bit PCM WAV container around mono float samples. */
function encodeWav(samples: Float32Array, sampleRate = TARGET_SAMPLE_RATE): Blob {
  const bytesPerSample = 2;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  const writeAscii = (offset: number, text: string) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };

  writeAscii(0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeAscii(8, "WAVE");
  writeAscii(12, "fmt ");
  view.setUint32(16, 16, true); // PCM header size
  view.setUint16(20, 1, true); // format: PCM
  view.setUint16(22, 1, true); // channels
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * bytesPerSample, true); // byte rate
  view.setUint16(32, bytesPerSample, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    // Clamp before scaling: values outside [-1, 1] would wrap and click.
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
    offset += bytesPerSample;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** Rough loudness of a live stream, for the recording meter. */
export function createLevelMeter(stream: MediaStream): {
  read: () => number;
  stop: () => void;
} {
  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  return {
    read: () => {
      analyser.getByteTimeDomainData(data);
      let peak = 0;
      for (const value of data) peak = Math.max(peak, Math.abs(value - 128));
      return Math.min(1, peak / 128);
    },
    stop: () => void context.close(),
  };
}
