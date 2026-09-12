export type InputSettings = {
  bands: string;
  scale: string;
  offset: string;
  layout: "CHW" | "HWC";
};

export type SatelliteImage = {
  id: string;
  url: string;
  name: string;
  width: number;
  height: number;
  sample: boolean;
};

export type InferenceResult = {
  url: string;
  download_url: string;
  width: number;
  height: number;
  step: number;
  elapsed_seconds: number;
  numeric_format: string;
};

export type BackendHealth = {
  status: "ready" | "unavailable";
  model: string;
  step: number | null;
  device: string;
  error: string | null;
};

type Job = {
  id: string;
  status: "queued" | "processing" | "complete" | "failed" | "cancelled";
  progress: number;
  message: string;
  error: string | null;
  result: InferenceResult | null;
};

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response;
  const controller = new AbortController();
  const forwardAbort = () => controller.abort();
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, 30000);
  options.signal?.addEventListener("abort", forwardAbort, { once: true });
  if (options.signal?.aborted) controller.abort();
  try {
    response = await fetch(`/api${path}`, {
      ...options,
      signal: controller.signal,
      cache: "no-store",
    });
  } catch (error) {
    if (timedOut)
      throw new Error(
        "The backend took too long to respond. Check the connection and try again.",
      );
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new Error(
      "The backend is unreachable. Start the RezX API and try again.",
    );
  } finally {
    window.clearTimeout(timer);
    options.signal?.removeEventListener("abort", forwardAbort);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(
      typeof body?.detail === "string"
        ? body.detail
        : "The backend could not complete the request. Please try again.",
    );
  }
  if (response.status === 204) return undefined as T;
  try {
    return (await response.json()) as T;
  } catch {
    throw new Error(
      "The inference API is not responding. Check the backend connection and try again.",
    );
  }
}

export const getHealth = (signal?: AbortSignal) =>
  request<BackendHealth>("/health", { signal });

export function releaseImage(image: SatelliteImage) {
  // A running job may briefly keep the image. The API also expires abandoned files.
  void request(`/images/${image.id}`, { method: "DELETE" }).catch(
    () => undefined,
  );
}

async function loadPreview(image: SatelliteImage) {
  try {
    const preview = new Image();
    preview.src = image.url;
    await preview.decode();
    return image;
  } catch {
    releaseImage(image);
    throw new Error(
      "The image preview could not be loaded. Check the backend connection and upload it again.",
    );
  }
}

export async function uploadImage(
  file: File,
  settings: InputSettings,
): Promise<SatelliteImage> {
  if (!/\.(tiff?|npy)$/i.test(file.name)) {
    throw new Error(
      "Upload a four-band TIFF or NumPy array. PNG/JPG images do not contain measured near-infrared data.",
    );
  }
  if (file.size > 20 * 1024 * 1024)
    throw new Error("Choose a file smaller than 20 MB.");
  const body = new FormData();
  body.append("file", file);
  for (const [key, value] of Object.entries(settings)) body.append(key, value);
  return loadPreview(
    await request<SatelliteImage>("/images", { method: "POST", body }),
  );
}

export async function getSample(): Promise<SatelliteImage> {
  return loadPreview(
    await request<SatelliteImage>("/images/sample", { method: "POST" }),
  );
}

function delay(ms: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("Cancelled", "AbortError"));
      return;
    }
    const abort = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", abort);
      reject(new DOMException("Cancelled", "AbortError"));
    };
    const timer = window.setTimeout(() => {
      signal.removeEventListener("abort", abort);
      resolve();
    }, ms);
    signal.addEventListener("abort", abort, { once: true });
  });
}

export async function superResolve(
  source: SatelliteImage,
  onProgress: (progress: number, message: string) => void,
  signal: AbortSignal,
): Promise<InferenceResult> {
  // Allow submission to return its job ID even if Cancel was pressed in transit.
  let job = await request<Job>("/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_id: source.id }),
  });
  const cancel = () => {
    void request(`/jobs/${job.id}`, { method: "DELETE" }).catch(
      () => undefined,
    );
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    while (true) {
      if (signal.aborted) {
        cancel();
        throw new DOMException("Cancelled", "AbortError");
      }
      onProgress(job.progress, job.message);
      if (job.status === "complete" && job.result) {
        if (
          job.result.width !== source.width * 4 ||
          job.result.height !== source.height * 4
        ) {
          throw new Error(
            "The backend returned unexpected image dimensions. Please check the model configuration.",
          );
        }
        const preview = new Image();
        preview.src = job.result.url;
        await preview.decode();
        if (signal.aborted) throw new DOMException("Cancelled", "AbortError");
        return job.result;
      }
      if (job.status === "failed")
        throw new Error(
          job.error || "Model processing failed. Please try again.",
        );
      if (job.status === "cancelled")
        throw new Error(
          "The previous request was cancelled. Click Super-resolve to start again.",
        );
      await delay(400, signal);
      job = await request<Job>(`/jobs/${job.id}`, { signal });
    }
  } finally {
    signal.removeEventListener("abort", cancel);
  }
}
