import { useCallback, useEffect, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  Check,
  ChevronRight,
  ChevronsLeftRight,
  Download,
  Earth,
  Expand,
  ImagePlus,
  Layers,
  Maximize2,
  MoveUpRight,
  Pause,
  Play,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import EarthScene from "./EarthScene";
import Atmosphere from "./Atmosphere";
import {
  superResolve,
  getSample,
  uploadImage,
  releaseImage,
  getHealth,
  type SatelliteImage,
  type InferenceResult,
  type InputSettings,
  type BackendHealth,
} from "./api";

type Stage = "orbit" | "diving" | "workspace";
type Phase = "empty" | "ready" | "processing" | "result";
type Modal = "how" | "about" | null;

function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

function Brand({ onClick }: { onClick: () => void }) {
  return (
    <button className="brand" onClick={onClick} aria-label="RezX home">
      <svg viewBox="0 0 38 38" aria-hidden="true">
        <circle cx="19" cy="19" r="10.5" />
        <ellipse cx="19" cy="19" rx="17" ry="6" transform="rotate(-38 19 19)" />
        <path d="M12 10C17 13 21 20 23 28" />
      </svg>
      <span>
        Rez<span className="brand-x">X</span>
      </span>
    </button>
  );
}

function InfoModal({
  type,
  onClose,
  onStart,
}: {
  type: Modal;
  onClose: () => void;
  onStart: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (type) dialog.current?.showModal();
    else dialog.current?.close();
  }, [type]);
  return (
    <dialog
      ref={dialog}
      className="info-dialog"
      aria-labelledby="info-title"
      onCancel={onClose}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="dialog-content">
        <button
          className="icon-button dialog-close"
          onClick={onClose}
          aria-label="Close dialog"
        >
          <X size={20} />
        </button>
        <span className="dialog-symbol">
          {type === "how" ? <Layers size={27} /> : <Earth size={27} />}
        </span>
        <p className="eyebrow">
          {type === "how"
            ? "From orbit to detail"
            : "A little more perspective"}
        </p>
        <h2 id="info-title">
          {type === "how"
            ? "Small image.\nBigger possibilities."
            : "Meet RezX."}
        </h2>
        {type === "how" ? (
          <>
            <div className="how-steps">
              <div>
                <span>01</span>
                <section>
                  <h3>Choose your perspective</h3>
                  <p>
                    Select an orbiting satellite, then upload a four-band TIFF
                    or NumPy array with blue, green, red and near-infrared data.
                  </p>
                </section>
              </div>
              <div>
                <span>02</span>
                <section>
                  <h3>Take a closer look</h3>
                  <p>
                    The trained model processes your image in tiles, using its
                    saved normalization and EMA weights.
                  </p>
                </section>
              </div>
              <div>
                <span>03</span>
                <section>
                  <h3>See the difference</h3>
                  <p>
                    Compare both views, then download the PNG, full four-band
                    data and inference report together.
                  </p>
                </section>
              </div>
            </div>
            <p className="dialog-note">
              Use prepared satellite data with the same band order, scale and
              offset used during training. The PNG display shares the original
              image’s color stretch.
            </p>
          </>
        ) : (
          <>
            <p className="about-copy">
              A new perspective on Earth observation. RezX brings a simple,
              thoughtful interface to satellite image super-resolution, one tiny
              world at a time.
            </p>
            <p className="about-copy">
              RezX runs the trained S2-EvidenceSR-4X model through a local
              inference API. Downloads preserve all four bands and the
              geographic footprint of GeoTIFF inputs.
            </p>
            <div className="about-detail">
              <Earth size={19} />
              <span>Built for a closer look at our planet.</span>
            </div>
          </>
        )}
        <button
          className="button button--dark"
          onClick={() => {
            onClose();
            onStart();
          }}
        >
          Explore the workspace
          <ArrowUpRight size={17} />
        </button>
      </div>
    </dialog>
  );
}

function Comparison({
  source,
  result,
  enabled,
}: {
  source: SatelliteImage;
  result: string;
  enabled: boolean;
}) {
  const [position, setPosition] = useState(50);
  return (
    <div
      className={`comparison ${enabled ? "comparison--enabled" : ""}`}
      style={{
        width: `min(100%, ${(390 * source.width) / source.height}px)`,
        aspectRatio: source.width / source.height,
      }}
    >
      <img
        className="result-image"
        src={result}
        alt="Super-resolved RGB view from the trained four-band model"
        draggable={false}
      />
      {enabled && (
        <>
          <div
            className="comparison-original"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            <img src={source.url} alt="Original image" draggable={false} />
          </div>
          <span className="image-label image-label--left">Original</span>
          <span className="image-label image-label--right">RezX</span>
          <div className="comparison-divider" style={{ left: `${position}%` }}>
            <span>
              <ChevronsLeftRight size={21} />
            </span>
          </div>
          <input
            type="range"
            min="0"
            max="100"
            value={position}
            onChange={(event) => setPosition(Number(event.target.value))}
            aria-label="Before and after comparison"
            aria-valuetext={`${position}% original image, ${100 - position}% preview`}
          />
        </>
      )}
    </div>
  );
}

export default function App() {
  const [stage, setStage] = useState<Stage>("orbit");
  const [phase, setPhase] = useState<Phase>("empty");
  const [satellite, setSatellite] = useState("01");
  const [paused, setPaused] = useState(false);
  const [modal, setModal] = useState<Modal>(null);
  const [source, setSource] = useState<SatelliteImage | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState("");
  const [reading, setReading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState(
    "Waiting for the model",
  );
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [connection, setConnection] = useState<
    "connecting" | "ready" | "unavailable" | "offline"
  >("connecting");
  const [settings, setSettings] = useState<InputSettings>({
    bands: "1,2,3,4",
    scale: "1",
    offset: "0",
    layout: "CHW",
  });
  const refreshHealth = useCallback(async (signal?: AbortSignal) => {
    try {
      const status = await getHealth(signal);
      setHealth(status);
      setConnection(status.status);
    } catch {
      if (!signal?.aborted) {
        setHealth(null);
        setConnection("offline");
      }
    }
  }, []);
  const [compare, setCompare] = useState(true);
  const [downloaded, setDownloaded] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const workspaceTitle = useRef<HTMLHeadingElement>(null);
  const orbitTitle = useRef<HTMLHeadingElement>(null);
  const previousStage = useRef<Stage>("orbit");
  const operation = useRef(0);
  const dragDepth = useRef(0);
  const reducedMotion = useReducedMotion();
  const enter = useCallback((id = "01") => {
    setSatellite(id);
    setStage((current) => (current === "orbit" ? "diving" : current));
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void refreshHealth(controller.signal);
    const interval = window.setInterval(() => {
      if (!document.hidden) void refreshHealth(controller.signal);
    }, 15000);
    return () => {
      controller.abort();
      window.clearInterval(interval);
    };
  }, [refreshHealth]);
  useEffect(() => {
    if (stage !== "diving") return;
    const timer = window.setTimeout(
      () => setStage("workspace"),
      reducedMotion ? 80 : 1900,
    );
    return () => window.clearTimeout(timer);
  }, [stage, reducedMotion]);
  useEffect(() => {
    if (stage === "workspace")
      workspaceTitle.current?.focus({ preventScroll: true });
    if (stage === "orbit" && previousStage.current !== "orbit")
      orbitTitle.current?.focus({ preventScroll: true });
    previousStage.current = stage;
  }, [stage]);
  useEffect(
    () => () => {
      if (source) releaseImage(source);
    },
    [source],
  );
  useEffect(
    () => () => {
      operation.current++;
    },
    [],
  );

  useEffect(() => {
    if (phase !== "processing" || !source) return;
    let disposed = false;
    const controller = new AbortController();
    superResolve(
      source,
      (value, message) => {
        if (!disposed) {
          setProgress(value);
          setProgressMessage(message);
        }
      },
      controller.signal,
    )
      .then((output) => {
        if (!disposed) {
          setResult(output);
          setPhase("result");
        }
      })
      .catch((err: unknown) => {
        if (!disposed) {
          setError(
            err instanceof Error
              ? err.message
              : "Model processing failed. Please try again.",
          );
          setPhase("ready");
          void refreshHealth();
        }
      });
    return () => {
      disposed = true;
      controller.abort();
    };
  }, [phase, source, refreshHealth]);

  async function chooseFile(file?: File) {
    if (!file || phase === "processing") return;
    const id = ++operation.current;
    setReading(true);
    setError("");
    try {
      const selected = await uploadImage(file, settings);
      if (id !== operation.current) {
        releaseImage(selected);
        return;
      }
      setSource(selected);
      setResult(null);
      setPhase("ready");
      setDownloaded(false);
    } catch (err) {
      if (id === operation.current)
        setError(
          err instanceof Error ? err.message : "This file could not be opened.",
        );
    } finally {
      if (id === operation.current) setReading(false);
    }
  }
  async function useSample() {
    const id = ++operation.current;
    setReading(true);
    setError("");
    try {
      const sample = await getSample();
      if (id !== operation.current) {
        releaseImage(sample);
        return;
      }
      setSource(sample);
      setResult(null);
      setPhase("ready");
      setDownloaded(false);
    } catch (err) {
      if (id === operation.current)
        setError(
          err instanceof Error
            ? err.message
            : "The sample could not be loaded.",
        );
    } finally {
      if (id === operation.current) setReading(false);
    }
  }
  function reset() {
    operation.current++;
    setReading(false);
    setSource(null);
    setResult(null);
    setPhase("empty");
    setProgress(0);
    setError("");
    setDownloaded(false);
    setCompare(true);
    if (fileInput.current) fileInput.current.value = "";
  }
  function goHome() {
    operation.current++;
    setReading(false);
    if (phase === "processing") setPhase("ready");
    setStage("orbit");
  }
  function download() {
    if (!result || !source) return;
    const anchor = document.createElement("a");
    anchor.href = result.download_url;
    anchor.download = `${source.name.replace(/\.[^.]+$/, "")}-rezx-4x.zip`;
    anchor.click();
    setDownloaded(true);
  }

  const isSpace = stage !== "workspace";
  return (
    <div
      className={`app ${isSpace ? "app--space" : "app--sky"} ${stage === "diving" ? "app--diving" : ""} ${paused || reducedMotion ? "motion-paused" : ""}`}
    >
      <a href="#main-content" className="skip-link">
        Skip to content
      </a>
      <header className="header">
        <Brand onClick={goHome} />
        <nav className="main-nav" aria-label="Main navigation">
          <button
            className={!modal ? "nav-link nav-link--active" : "nav-link"}
            onClick={goHome}
          >
            Explore
          </button>
          <button
            className={
              modal === "how" ? "nav-link nav-link--active" : "nav-link"
            }
            onClick={() => setModal("how")}
          >
            How it works
          </button>
          <button
            className={
              modal === "about" ? "nav-link nav-link--active" : "nav-link"
            }
            onClick={() => setModal("about")}
          >
            About RezX
          </button>
        </nav>
        {isSpace ? (
          <button className="header-action" onClick={() => enter()}>
            Open workspace
            <ArrowUpRight size={15} />
          </button>
        ) : (
          <button className="header-action" onClick={goHome}>
            <ArrowLeft size={14} />
            Back to orbit
          </button>
        )}
      </header>

      <main id="main-content" tabIndex={-1}>
        {isSpace && (
          <section className="landing" aria-label="Explore RezX">
            <div className="hero-copy">
              <div className="intro-label">
                <span className="status-dot" />A new perspective on Earth
              </div>
              <h1 ref={orbitTitle} tabIndex={-1}>
                A little closer.
                <br />A lot clearer.
              </h1>
              <p className="hero-description">
                Our planet has more to show you.
                <br className="desktop-break" /> Turn satellite imagery into a
                world of detail
                <br className="desktop-break" /> with 4× super-resolution.
              </p>
              <button
                className="button button--mint launch-button"
                onClick={() => enter()}
              >
                Let’s take a closer look
                <ArrowUpRight size={18} />
              </button>
              <div className="satellite-hint">
                <span className="hint-line" />
                or choose a satellite to begin
              </div>
              <div className="hero-specs">
                <div>
                  <strong>4×</strong>
                  <span>Resolution</span>
                </div>
                <i />
                <div>
                  <strong>One planet.</strong>
                  <span>Infinite perspectives.</span>
                </div>
              </div>
            </div>
            <div className="planet-area">
              <div className="planet-halo" />
              <div className="orbit-caption">
                <span className="tiny-cross">+</span>
                <span>
                  YOUR NEXT PERSPECTIVE
                  <br />
                  <b>STARTS UP HERE.</b>
                </span>
              </div>
              <EarthScene
                diving={stage === "diving"}
                paused={paused}
                reducedMotion={reducedMotion}
                onEnter={enter}
              />
              <div className="earth-caption">
                <span className="earth-caption-icon">
                  <Earth size={17} />
                </span>
                <div>
                  <strong>Earth, in a little more detail.</strong>
                  <span>A familiar world. A fresh perspective.</span>
                </div>
              </div>
              <div className="scene-controls">
                <span>
                  <i />
                  In orbit
                </span>
                <button
                  onClick={() => setPaused(!paused)}
                  aria-label={
                    paused
                      ? "Resume ambient animation"
                      : "Pause ambient animation"
                  }
                  aria-pressed={paused}
                >
                  {paused ? <Play size={13} /> : <Pause size={13} />}
                </button>
              </div>
            </div>
            <div className="landing-bottom">
              <div className="resolution-preview">
                <span className="pixel-symbol pixel-symbol--before" />
                <ArrowRight size={14} />
                <span className="pixel-symbol pixel-symbol--after" />
                <p>
                  A small image.
                  <br />
                  <strong>A bigger picture.</strong>
                </p>
              </div>
              <button className="learn-link" onClick={() => setModal("how")}>
                Discover how it works
                <ArrowDown size={15} />
              </button>
            </div>
          </section>
        )}

        {stage === "diving" && (
          <div className="dive-overlay" aria-live="polite">
            <Atmosphere active />
            <span>Finding a fresh perspective…</span>
          </div>
        )}

        {stage === "workspace" && (
          <section
            className={`workspace workspace--${phase}`}
            aria-label="Image workspace"
          >
            <Atmosphere active reveal={phase === "result"} />
            <div className="workspace-inner">
              <div className="workspace-breadcrumb">
                <span className="status-dot" />
                Satellite {satellite}
                <ChevronRight size={13} />
                <span>Image workspace</span>
                <span
                  className={`preview-badge connection--${connection}`}
                  role="status"
                >
                  {connection === "ready"
                    ? "Model connected"
                    : connection === "connecting"
                      ? "Connecting…"
                      : connection === "offline"
                        ? "Backend offline"
                        : "Model unavailable"}
                </span>
              </div>
              <div className="workspace-heading">
                <p className="eyebrow">
                  {phase === "result"
                    ? "A fresh perspective"
                    : phase === "processing"
                      ? "A little closer, frame by frame"
                      : "Down to the details"}
                </p>
                <h1 ref={workspaceTitle} tabIndex={-1}>
                  {phase === "result"
                    ? "The bigger picture."
                    : phase === "processing"
                      ? "Good things take a moment."
                      : "Your world. In more detail."}
                </h1>
                <p>
                  {phase === "result"
                    ? "Slide between perspectives. Take a closer look."
                    : phase === "processing"
                      ? "Stay a while. Your satellite image is being super-resolved."
                      : "Bring an image. Discover a new perspective."}
                </p>
              </div>
              <input
                ref={fileInput}
                type="file"
                accept=".tif,.tiff,.npy"
                className="file-input"
                aria-label="Choose an image to upload"
                onChange={(event) => {
                  void chooseFile(event.target.files?.[0]);
                  event.target.value = "";
                }}
              />
              {connection !== "ready" && connection !== "connecting" && (
                <div className="connection-notice" role="alert">
                  <p>
                    {health?.error ||
                      "The backend is offline. Start the RezX API to upload and process satellite images."}
                  </p>
                  <button
                    className="text-button"
                    onClick={() => void refreshHealth()}
                  >
                    Check connection
                  </button>
                </div>
              )}
              {phase === "empty" && (
                <>
                  <details className="input-settings">
                    <summary>
                      Input settings <span>B2 · B3 · B4 · B8</span>
                    </summary>
                    <p>
                      Match the band order and decoding used during training.
                      Defaults are prepared floating-point B/G/R/NIR data.
                    </p>
                    <div className="settings-grid">
                      <label>
                        Band numbers (B/G/R/NIR)
                        <input
                          value={settings.bands}
                          onChange={(event) =>
                            setSettings({
                              ...settings,
                              bands: event.target.value,
                            })
                          }
                          placeholder="1,2,3,4"
                        />
                      </label>
                      <label>
                        NumPy layout
                        <select
                          value={settings.layout}
                          onChange={(event) =>
                            setSettings({
                              ...settings,
                              layout: event.target.value as "CHW" | "HWC",
                            })
                          }
                        >
                          <option value="CHW">Bands × height × width</option>
                          <option value="HWC">Height × width × bands</option>
                        </select>
                      </label>
                      <label>
                        Scale
                        <input
                          value={settings.scale}
                          onChange={(event) =>
                            setSettings({
                              ...settings,
                              scale: event.target.value,
                            })
                          }
                          placeholder="1"
                        />
                      </label>
                      <label>
                        Offset
                        <input
                          value={settings.offset}
                          onChange={(event) =>
                            setSettings({
                              ...settings,
                              offset: event.target.value,
                            })
                          }
                          placeholder="0"
                        />
                      </label>
                    </div>
                    <p>
                      Decoded value = stored value × scale + offset. For raw DN,
                      enter your training decoder’s values; no automatic scaling
                      is applied.
                    </p>
                  </details>

                  <div
                    className={`upload-panel glass-panel ${dragging ? "upload-panel--dragging" : ""}`}
                    onDragEnter={(event) => {
                      event.preventDefault();
                      dragDepth.current++;
                      setDragging(true);
                    }}
                    onDragOver={(event) => {
                      event.preventDefault();
                      event.dataTransfer.dropEffect = "copy";
                    }}
                    onDragLeave={(event) => {
                      event.preventDefault();
                      dragDepth.current--;
                      if (dragDepth.current <= 0) setDragging(false);
                    }}
                    onDrop={(event) => {
                      event.preventDefault();
                      dragDepth.current = 0;
                      setDragging(false);
                      if (event.dataTransfer.files.length > 1)
                        setError("Choose one image at a time.");
                      else void chooseFile(event.dataTransfer.files[0]);
                    }}
                  >
                    <div className="upload-illustration">
                      <div className="mini-image mini-image--back">
                        <Layers size={33} />
                      </div>
                      <div className="mini-image mini-image--front">
                        <ImagePlus size={34} strokeWidth={1.3} />
                        <span className="mini-plus">+</span>
                      </div>
                    </div>
                    <button
                      className="upload-button"
                      onClick={() => fileInput.current?.click()}
                      disabled={reading || connection !== "ready"}
                    >
                      <Upload size={20} strokeWidth={1.6} />
                      {reading ? "Opening image…" : "Upload image"}
                      <ArrowUpRight size={18} />
                    </button>
                    <p className="drop-hint">
                      {dragging
                        ? "Drop your image here"
                        : "or drop your image into this little patch of sky"}
                    </p>
                    <span className="upload-formats">
                      4-band TIFF or NumPy · Up to 20 MB · Max 1024 × 1024
                    </span>
                  </div>
                  <button
                    className="sample-button"
                    onClick={() => void useSample()}
                    disabled={reading || connection !== "ready"}
                  >
                    <span className="sample-thumbnail" />
                    <span>
                      No image on hand? <strong>Try a four-band sample</strong>
                    </span>
                    <ArrowRight size={15} />
                  </button>
                </>
              )}
              {(phase === "ready" || phase === "processing") && source && (
                <div className="image-panel glass-panel">
                  <div className="image-panel-top">
                    <div>
                      <span className="status-dot" />
                      <span>
                        {phase === "processing"
                          ? "Running the model"
                          : "Ready for a closer look"}
                      </span>
                    </div>
                    {phase === "ready" && (
                      <button
                        className="icon-button"
                        onClick={reset}
                        aria-label="Remove selected image"
                      >
                        <X size={17} />
                      </button>
                    )}
                  </div>
                  <div className="selected-image">
                    <img
                      src={source.url}
                      alt={`Selected image: ${source.name}`}
                    />
                    {phase === "processing" && <div className="image-scan" />}
                  </div>
                  {source.sample && (
                    <p className="sample-notice">
                      Synthetic four-band sample · Runs the real model;
                      demonstrates the workflow only.
                    </p>
                  )}
                  <div className="image-metadata">
                    <span title={source.name}>{source.name}</span>
                    <span>
                      {source.width} × {source.height}
                      <ArrowRight size={12} />
                      {source.width * 4} × {source.height * 4}
                    </span>
                  </div>
                  {phase === "ready" ? (
                    <button
                      className="button button--dark resolve-button"
                      disabled={connection !== "ready"}
                      onClick={() => {
                        setError("");
                        setProgress(0);
                        setProgressMessage(
                          "Submitting your image to the model",
                        );
                        setPhase("processing");
                      }}
                    >
                      <Sparkles size={17} />
                      Super-resolve
                      <MoveUpRight size={17} />
                    </button>
                  ) : (
                    <div className="processing-status" aria-live="polite">
                      <div>
                        <span>SUPER-RESOLVING</span>
                        <span>{Math.round(progress)}%</span>
                      </div>
                      <div
                        className="progress-track"
                        role="progressbar"
                        aria-label="Super-resolving image"
                        aria-valuemin={0}
                        aria-valuemax={100}
                        aria-valuenow={Math.round(progress)}
                      >
                        <span style={{ width: `${progress}%` }} />
                      </div>
                      <p>{progressMessage}</p>
                      <button
                        className="text-button"
                        onClick={() => setPhase("ready")}
                      >
                        Cancel
                      </button>
                    </div>
                  )}
                </div>
              )}
              {phase === "result" && source && result && (
                <div className="result-panel glass-panel">
                  <div className="image-panel-top">
                    <div>
                      <span className="complete-check">
                        <Check size={12} />
                      </span>
                      <span>Super-resolution complete</span>
                    </div>
                    <span className="result-scale">
                      <Maximize2 size={13} />
                      4×
                    </span>
                  </div>
                  <Comparison
                    source={source}
                    result={result.url}
                    enabled={compare}
                  />
                  <div className="result-dimensions">
                    <span>
                      {source.width} × {source.height}
                    </span>
                    <ArrowRight size={17} />
                    <strong>
                      {source.width * 4} × {source.height * 4}
                    </strong>
                    <span className="format-tag">PNG + 4 BANDS</span>
                  </div>
                  <p className="result-provenance">
                    Trained model · Step {result.step.toLocaleString()} ·{" "}
                    {result.elapsed_seconds.toFixed(1)}s ·{" "}
                    {result.numeric_format}
                  </p>
                  {source.sample && (
                    <p className="sample-notice">Synthetic sample result</p>
                  )}
                  <div className="result-actions">
                    <button
                      className={`button compare-button ${compare ? "is-selected" : ""}`}
                      aria-pressed={compare}
                      onClick={() => setCompare(!compare)}
                    >
                      <ScanLine size={17} />
                      Compare
                    </button>
                    <button
                      className="button button--dark download-button"
                      onClick={download}
                    >
                      {downloaded ? (
                        <Check size={17} />
                      ) : (
                        <Download size={17} />
                      )}
                      Download
                    </button>
                    <button className="button new-image-button" onClick={reset}>
                      <ImagePlus size={17} />
                      New image
                    </button>
                  </div>
                  {downloaded && (
                    <p className="download-status" role="status">
                      Your result ZIP download has started.
                    </p>
                  )}
                </div>
              )}
              {error && (
                <div className="error-message" role="alert">
                  <span>{error}</span>
                  <button
                    className="icon-button"
                    aria-label="Dismiss error"
                    onClick={() => setError("")}
                  >
                    <X size={16} />
                  </button>
                </div>
              )}
              <div className="workspace-note">
                <ShieldCheck size={14} />
                <p>
                  Processed by your connected RezX backend.
                  <span>
                    Downloads include the RGB preview, four-band data and
                    processing report.
                  </span>
                </p>
              </div>
              {phase === "empty" && (
                <div className="workspace-features">
                  <span>
                    <Expand size={14} />
                    4× the dimensions
                  </span>
                  <span>
                    <Layers size={14} />
                    One simple workflow
                  </span>
                  <span>
                    <Download size={14} />
                    Yours to explore
                  </span>
                </div>
              )}
            </div>
          </section>
        )}
      </main>
      <footer className="footer">
        <span>Earth observation, reimagined.</span>
        <div>
          <span className="footer-orbit" />
          <span>A little closer to our planet.</span>
        </div>
        <span>© {new Date().getFullYear()} RezX</span>
      </footer>
      <InfoModal
        type={modal}
        onClose={() => setModal(null)}
        onStart={() => enter()}
      />
    </div>
  );
}
