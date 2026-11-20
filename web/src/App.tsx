import { useEffect, useState } from "react";

const API = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

type Study = { id: string; modalities: string[] };
type Analysis = {
  analysis_id: string;
  created_at?: string;
  label: string;
  score: number;
  overlay_png_base64: string;
  model_version: string;
  disclaimer: string;
};
type Segmentation = {
  segmentation_id: string;
  overlay_png_base64: string;
  slice_index: number;
  slice_count: number;
  voxel_counts: Record<string, number>;
  model_version: string;
  disclaimer: string;
};
type Review = {
  id: string;
  reviewer: string;
  decision: "approve" | "reject";
  note: string;
  created_at: string;
};

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `Request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export default function App() {
  const [studies, setStudies] = useState<Study[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [segmentation, setSegmentation] = useState<Segmentation | null>(null);
  const [segmentationReady, setSegmentationReady] = useState(false);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const selectedStudy = studies.find((study) => study.id === selectedId);
  const supportsClassifier =
    selectedStudy !== undefined &&
    selectedStudy.modalities.length > 0 &&
    selectedStudy.modalities.every((modality) => ["CR", "DX"].includes(modality));
  const supportsSegmentation = selectedStudy?.modalities.includes("CT") ?? false;

  const refreshStudies = async () => {
    try {
      setStudies(await request<Study[]>("/api/studies"));
      setError("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not load studies");
    }
  };

  useEffect(() => {
    void refreshStudies();
    request<{ segmentation_ready: boolean }>("/api/health")
      .then((health) => setSegmentationReady(health.segmentation_ready))
      .catch((reason: unknown) =>
        setError(reason instanceof Error ? reason.message : "Could not check model readiness"),
      );
  }, []);

  useEffect(() => {
    setAnalysis(null);
    setSegmentation(null);
    setReviews([]);
    setError("");
    if (!selectedId || !supportsClassifier) return;
    let active = true;
    const refreshAnalysis = async () => {
      try {
        const saved = await request<Analysis | null>(
          `/api/studies/${selectedId}/analysis`,
        );
        if (active) setAnalysis(saved);
      } catch (reason) {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load saved analysis");
        }
      }
    };
    void refreshAnalysis();
    const analysisTimer = window.setInterval(() => void refreshAnalysis(), 5000);
    request<Review[]>(`/api/audit?study_id=${encodeURIComponent(selectedId)}`)
      .then((savedReviews) => {
        if (active) setReviews(savedReviews);
      })
      .catch((reason: unknown) => {
        if (active) {
          setError(reason instanceof Error ? reason.message : "Could not load audit log");
        }
      });
    return () => {
      active = false;
      window.clearInterval(analysisTimer);
    };
  }, [selectedId, supportsClassifier]);

  const runAnalysis = async () => {
    if (!selectedId || !supportsClassifier) return;
    setBusy(true);
    setError("");
    try {
      setAnalysis(
        await request<Analysis>(`/api/studies/${selectedId}/analyze`, { method: "POST" }),
      );
    } catch (reason) {
      setAnalysis(null);
      setError(reason instanceof Error ? reason.message : "Analysis failed");
    } finally {
      setBusy(false);
    }
  };

  const runSegmentation = async () => {
    if (!selectedId || !supportsSegmentation) return;
    setBusy(true);
    setError("");
    try {
      setSegmentation(
        await request<Segmentation>(`/api/studies/${selectedId}/segment`, { method: "POST" }),
      );
    } catch (reason) {
      setSegmentation(null);
      setError(reason instanceof Error ? reason.message : "3D segmentation failed");
    } finally {
      setBusy(false);
    }
  };

  const submitReview = async (decision: "approve" | "reject") => {
    if (!selectedId || !supportsClassifier || !analysis || !reviewer.trim()) return;
    setBusy(true);
    setError("");
    try {
      const saved = await request<Review>(`/api/studies/${selectedId}/review`, {
        method: "POST",
        body: JSON.stringify({
          analysis_id: analysis.analysis_id,
          reviewer: reviewer.trim(),
          decision,
          note,
        }),
      });
      setReviews((previous) => [...previous, saved]);
      setNote("");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save review");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">IMAGING REVIEW WORKSPACE</p>
          <h1>Study review</h1>
        </div>
        <div className="notice" role="alert">
          <strong>For education only — not for clinical use.</strong>
          <span>Do not use model output for patient care.</span>
        </div>
      </header>

      <div className="layout">
        <aside className="panel study-list">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">ORTHANC</p>
              <h2>Studies</h2>
            </div>
            <button className="quiet-button" onClick={() => void refreshStudies()}>
              Refresh
            </button>
          </div>
          {studies.length === 0 ? (
            <p className="muted">No studies found. Import a permitted DICOM study into Orthanc.</p>
          ) : (
            <ul>
              {studies.map((study) => (
                <li key={study.id}>
                  <button
                    className={`study-row ${selectedId === study.id ? "selected" : ""}`}
                    onClick={() => setSelectedId(study.id)}
                  >
                    <span className="status-dot" />
                    <span className="study-id">{study.id}</span>
                    <span className="study-modality">{study.modalities.join("/") || "Unknown"}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="content">
          {error && <div className="error" role="alert">{error}</div>}
          <section className="panel viewer-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">DICOM PREVIEW</p>
                <h2>{selectedId ? `Study ${selectedId}` : "Select a study"}</h2>
              </div>
              <div className="model-actions">
                <button
                  className="primary-button"
                  disabled={!supportsClassifier || busy}
                  onClick={() => void runAnalysis()}
                  title={supportsClassifier ? "Classify a CR/DX chest radiograph" : "Classifier supports CR/DX studies only"}
                >
                  {busy ? "Working…" : "Run analysis"}
                </button>
                <button
                  className="secondary-button"
                  disabled={!supportsSegmentation || busy || !segmentationReady}
                  onClick={() => void runSegmentation()}
                  title={!segmentationReady ? "Configure a MONAI 3D checkpoint" : "Requires a CT series with at least 8 slices"}
                >
                  {segmentationReady ? "Run 3D segmentation" : "3D model unavailable"}
                </button>
              </div>
            </div>
            <div className="viewer">
              {selectedId ? (
                <>
                  <img
                    className="scan-image"
                    src={`${API}/api/studies/${selectedId}/preview`}
                    alt="Preview of the selected DICOM study"
                  />
                  {analysis && supportsClassifier && (
                    <img
                      className="heatmap"
                      src={`data:image/png;base64,${analysis.overlay_png_base64}`}
                      alt="Grad-CAM research visualization overlay"
                    />
                  )}
                  {(!analysis || !supportsClassifier) && (
                    <span className="viewer-hint">Preview only · no compatible AI overlay</span>
                  )}
                </>
              ) : (
                <div className="empty-viewer">
                  <span className="scan-icon">◫</span>
                  <p>Select a study from the list to view its preview.</p>
                </div>
              )}
            </div>
            {segmentation && (
              <div className="segmentation-output">
                <div>
                  <p className="eyebrow">MONAI 3D U-NET · CENTRAL MODEL-SPACE SLICE</p>
                  <p className="muted">
                    Slice {segmentation.slice_index + 1} of {segmentation.slice_count} ·{" "}
                    {segmentation.model_version}
                  </p>
                </div>
                <img
                  src={`data:image/png;base64,${segmentation.overlay_png_base64}`}
                  alt="Educational segmentation output overlaid on the central resampled CT slice"
                />
                <ul>
                  {Object.entries(segmentation.voxel_counts).map(([label, count]) => (
                    <li key={label}>{label}: {count.toLocaleString()} predicted voxels</li>
                  ))}
                </ul>
                <p className="fine-print">{segmentation.disclaimer}</p>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">MODEL OUTPUT</p>
                <h2>Research result</h2>
              </div>
              <span className={`pill ${analysis && supportsClassifier ? "ready" : ""}`}>
                {analysis && supportsClassifier ? "Unvalidated" : "Not run"}
              </span>
            </div>
            {analysis && supportsClassifier ? (
              <>
                <div className="result-grid">
                  <div><span className="muted">Top label</span><strong>{analysis.label}</strong></div>
                  <div><span className="muted">Uncalibrated score</span><strong>{(analysis.score * 100).toFixed(1)}%</strong></div>
                  <div><span className="muted">Model version</span><strong>{analysis.model_version}</strong></div>
                </div>
                <p className="fine-print">
                  Grad-CAM is a coarse visualization and does not establish that a finding is
                  present or explain the model reliably.
                </p>
              </>
            ) : (
              <p className="muted">
                {selectedId && !supportsClassifier
                  ? "The classifier supports CR/DX chest radiographs only. Use 3D segmentation for CT."
                  : "Analysis requires a configured DenseNet checkpoint. No result is inferred otherwise."}
              </p>
            )}
          </section>

          <section className="panel review-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">HUMAN REVIEW</p>
                <h2>Decision and audit trail</h2>
              </div>
              <span className="count">{reviews.length} decisions</span>
            </div>
            <div className="review-form">
              <label>
                Reviewer identifier
                <input value={reviewer} onChange={(event) => setReviewer(event.target.value)} placeholder="Enter reviewer ID" maxLength={120} />
              </label>
              <label>
                Note (optional)
                <input value={note} onChange={(event) => setNote(event.target.value)} placeholder="Add review context" maxLength={2000} />
              </label>
            </div>
            <div className="decision-actions">
              <button className="approve-button" disabled={!supportsClassifier || !analysis || !reviewer.trim() || busy} onClick={() => void submitReview("approve")}>Approve</button>
              <button className="reject-button" disabled={!supportsClassifier || !analysis || !reviewer.trim() || busy} onClick={() => void submitReview("reject")}>Reject</button>
            </div>
            {reviews.length > 0 && (
              <ul className="audit-list">
                {reviews.map((review) => (
                  <li key={review.id}>
                    <span className={`decision-tag ${review.decision}`}>{review.decision}</span>
                    <span><strong>{review.reviewer}</strong>{review.note && ` · ${review.note}`}</span>
                    <time>{new Date(review.created_at).toLocaleString()}</time>
                  </li>
                ))}
              </ul>
            )}
          </section>
        </section>
      </div>
      <footer>Educational prototype · Research outputs are not clinical advice.</footer>
    </main>
  );
}
