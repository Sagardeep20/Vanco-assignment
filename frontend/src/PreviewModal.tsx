import { useCallback, useEffect, useState } from "react";

type AssetDetail = {
  id: string;
  filename: string;
  original_path: string;
  relative_path: string;
  file_type: string;
  mime_type: string | null;
  file_size: number | null;
  status: string;
  error_message: string | null;
  description: string | null;
  extracted_text: string | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
};

type Props = {
  assetId: string;
  apiUrl: string;
  onClose: () => void;
};

function formatFileSize(bytes: number | null): string {
  if (bytes == null) return "unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function PreviewModal({ assetId, apiUrl, onClose }: Props) {
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [fileMissing, setFileMissing] = useState(false);
  const [mediaError, setMediaError] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [opening, setOpening] = useState(false);

  const fileUrl = `${apiUrl}/api/assets/${assetId}/file`;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setFileMissing(false);
    setMediaError("");
    setAsset(null);
    (async () => {
      try {
        const detailRes = await fetch(`${apiUrl}/api/assets/${assetId}`, {
          signal: controller.signal,
        });
        if (detailRes.status === 404) throw new Error("Asset not found.");
        if (!detailRes.ok) throw new Error(`Could not load asset (${detailRes.status}).`);
        const detail: AssetDetail = await detailRes.json();
        // Existence check: HEAD is cheap (headers only, no body).
        // Some servers/proxies reject HEAD with 405/501 even though GET
        // works — in that case fall through and let the GET-based
        // <img>/<video>/<iframe> below perform the real check.
        const headRes = await fetch(fileUrl, {
          method: "HEAD",
          signal: controller.signal,
        });
        if (headRes.status === 404) {
          setAsset(detail);
          setFileMissing(true);
        } else if (headRes.status === 405 || headRes.status === 501) {
          setAsset(detail);
        } else if (!headRes.ok) {
          setAsset(detail);
          setMediaError(`Preview unavailable (server returned ${headRes.status}).`);
        } else {
          setAsset(detail);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setError(err instanceof Error ? err.message : "Could not load preview.");
      } finally {
        setLoading(false);
      }
    })();
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [assetId, apiUrl]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = previous;
    };
  }, [onClose]);

  const copyPath = useCallback(async (path: string) => {
    setActionMessage("");
    try {
      await navigator.clipboard.writeText(path);
      setActionMessage("Original path copied to clipboard.");
    } catch {
      setActionMessage("Copy failed — select the path manually.");
    }
  }, []);

  async function openOriginal() {
    setOpening(true);
    setActionMessage("");
    try {
      const res = await fetch(`${apiUrl}/api/assets/${assetId}/open`, { method: "POST" });
      if (res.status === 404) {
        setActionMessage("Original file no longer exists.");
        return;
      }
      const body = await res.json();
      setActionMessage(body.message ?? (body.opened ? "Opened." : "Could not open."));
    } catch (err) {
      setActionMessage(err instanceof Error ? err.message : "Could not open the file.");
    } finally {
      setOpening(false);
    }
  }

  function renderMedia(detail: AssetDetail) {
    if (fileMissing) {
      return (
        <p className="rounded-md border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          The indexed file is missing from the dataset directory. Metadata below is the
          last known state.
        </p>
      );
    }
    if (mediaError) {
      return (
        <p className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
          {mediaError}
        </p>
      );
    }
    if (detail.status === "unsupported") {
      return (
        <p className="rounded-md border bg-gray-50 p-4 text-sm text-gray-600">
          No inline preview for this file type. You can still open or copy the original
          below.
        </p>
      );
    }
    if (detail.file_type === "image") {
      return (
        <img
          src={fileUrl}
          alt={detail.filename}
          onError={() => setMediaError("The image could not be loaded.")}
          className="max-h-[50vh] w-full rounded-md border object-contain bg-gray-50"
        />
      );
    }
    if (detail.file_type === "video") {
      return (
        <video
          src={fileUrl}
          controls
          preload="metadata"
          onError={() => setMediaError("The video could not be loaded.")}
          className="max-h-[50vh] w-full rounded-md border bg-black"
        >
          Your browser cannot play this video.
        </video>
      );
    }
    if (detail.file_type === "pdf") {
      return (
        <div>
          <iframe
            src={fileUrl}
            title={detail.filename}
            className="h-[50vh] w-full rounded-md border bg-white"
          />
          <a
            href={fileUrl}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-block text-xs text-blue-600 hover:underline"
          >
            Open PDF in a new tab
          </a>
        </div>
      );
    }
    return (
      <p className="rounded-md border bg-gray-50 p-4 text-sm text-gray-600">
        No inline preview for this file type.
      </p>
    );
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 p-4"
      onClick={onClose}
      role="presentation"
    >
      <div
        className="max-h-[90vh] w-full max-w-3xl overflow-y-auto rounded-2xl border border-slate-200 bg-white p-5 shadow-2xl sm:p-6"
        onClick={(e) => e.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="Asset preview"
      >
        <div className="flex items-start justify-between gap-4 border-b border-slate-100 pb-4">
          <div className="min-w-0">
            <h2 className="truncate text-base font-semibold tracking-tight">{asset?.filename ?? "Preview"}</h2>
            {asset && (
              <p className="mt-0.5 truncate text-xs text-slate-500">{asset.relative_path}</p>
            )}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close preview"
            className="shrink-0 rounded-lg border border-slate-200 px-2.5 py-1.5 text-sm text-slate-500 transition hover:bg-slate-50 hover:text-slate-700"
          >
            ✕
          </button>
        </div>

        {loading && <p className="mt-4 text-sm text-gray-500">Loading preview…</p>}
        {!loading && error && (
          <p className="mt-4 rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-800">
            {error}
          </p>
        )}
        {!loading && !error && asset && (
          <div className="mt-4 space-y-4">
            {asset.status === "failed" && (
              <p className="rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-800">
                Processing failed{asset.error_message ? `: ${asset.error_message}` : "."}
              </p>
            )}
            {renderMedia(asset)}

            <dl className="grid grid-cols-2 gap-2.5 rounded-xl border border-slate-200 bg-slate-50/70 p-3.5 text-xs sm:grid-cols-4">
              <div>
                <dt className="text-gray-500">File type</dt>
                <dd className="mt-0.5 font-medium">{asset.file_type}</dd>
              </div>
              <div>
                <dt className="text-gray-500">File size</dt>
                <dd className="mt-0.5 font-medium">{formatFileSize(asset.file_size)}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Relative path</dt>
                <dd className="mt-0.5 break-all font-medium">{asset.relative_path}</dd>
              </div>
              <div>
                <dt className="text-gray-500">Processing status</dt>
                <dd className="mt-0.5 font-medium">{asset.status}</dd>
              </div>
            </dl>

            {asset.description && (
              <div className="text-xs">
                <p className="text-gray-500">Description</p>
                <p className="mt-0.5 text-gray-700">{asset.description}</p>
              </div>
            )}
            {asset.extracted_text && (
              <div className="text-xs">
                <p className="text-gray-500">Extracted text</p>
                <p className="mt-0.5 max-h-32 overflow-y-auto whitespace-pre-wrap rounded border bg-gray-50 p-2 text-gray-700">
                  {asset.extracted_text.slice(0, 1500)}
                  {asset.extracted_text.length > 1500 ? "…" : ""}
                </p>
              </div>
            )}

            <div className="text-xs">
              <p className="text-gray-500">Original path</p>
              <div className="mt-1 flex flex-col gap-2 sm:flex-row">
                <input
                  type="text"
                  readOnly
                  value={asset.original_path}
                  aria-label="Original file path"
                  className="w-full flex-1 rounded-md border bg-gray-50 px-2 py-1.5 font-mono text-[11px]"
                />
                <div className="flex shrink-0 gap-2">
                  <button
                    type="button"
                    onClick={() => void copyPath(asset.original_path)}
                    className="rounded-md border px-3 py-1.5 text-xs hover:bg-gray-50"
                  >
                    Copy path
                  </button>
                  <button
                    type="button"
                    onClick={openOriginal}
                    disabled={opening || fileMissing}
                    title="Ask the server host to open the file with its default app (works when server and desktop are the same machine)"
                    className="rounded-md bg-blue-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
                  >
                    {opening ? "Opening…" : "Open original"}
                  </button>
                </div>
              </div>
              {actionMessage && (
                <p className="mt-1 text-gray-600">{actionMessage}</p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
