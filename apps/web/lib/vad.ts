/**
 * Voice-activity detection on the same AnalyserNode style as the level meter.
 *
 * Speech starts when RMS stays above a threshold; it ends after a stretch of
 * quiet (2 s by default) so a brief pause mid-sentence does not cut the turn.
 * The recorder still owns capture and the 30 s hard cap.
 */

export const END_SILENCE_MS = 2000;

export type VadHandle = {
  read: () => number;
  stop: () => void;
};

export function createVad(
  stream: MediaStream,
  options: {
    threshold?: number;
    silenceMs?: number;
    onSpeechStart?: () => void;
    onSpeechEnd?: () => void;
    onLevel?: (level: number) => void;
  } = {},
): VadHandle {
  const threshold = options.threshold ?? 0.045;
  const silenceMs = options.silenceMs ?? END_SILENCE_MS;

  const context = new AudioContext();
  const analyser = context.createAnalyser();
  analyser.fftSize = 512;
  context.createMediaStreamSource(stream).connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);

  let speaking = false;
  let silenceStarted: number | null = null;
  let raf = 0;
  let stopped = false;

  const readRms = () => {
    analyser.getByteTimeDomainData(data);
    let sum = 0;
    for (const value of data) {
      const sample = (value - 128) / 128;
      sum += sample * sample;
    }
    return Math.sqrt(sum / data.length);
  };

  const tick = () => {
    if (stopped) return;
    const level = readRms();
    options.onLevel?.(Math.min(1, level * 4));

    if (level >= threshold) {
      if (!speaking) {
        speaking = true;
        options.onSpeechStart?.();
      }
      silenceStarted = null;
    } else if (speaking) {
      if (silenceStarted === null) silenceStarted = performance.now();
      else if (performance.now() - silenceStarted >= silenceMs) {
        speaking = false;
        silenceStarted = null;
        options.onSpeechEnd?.();
      }
    }

    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return {
    read: readRms,
    stop: () => {
      stopped = true;
      cancelAnimationFrame(raf);
      void context.close();
    },
  };
}
