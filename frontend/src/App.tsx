import { useCallback, useEffect, useRef, useState } from "react";
import PreviewModal from "./PreviewModal.tsx";

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";
const SEARCH_DEBOUNCE_MS = 400;
const SEARCH_LIMIT = 20;
const PREVIEW_LENGTH = 220;

type HealthResponse = {
  status: string;
  service?: string;
};

type Stats = {
  total: number;
  images: number;
  videos: number;
  documents: number;
  pending: number;
  completed: number;
  duplicates: number;
  failed: number;
  unsupported: number;
};

type Asset = {
  id: string;
  filename: string;
  relative_path: string;
  file_type: string;
  file_size: number | null;
  status: string;
  description: string | null;
  extracted_text: string | null;
  width: number | null;
  height: number | null;
  duration_seconds: number | null;
};

type SearchResult = {
  asset: Asset;
  score: number | null;
};

type FileFilter = "all" | "image" | "video" | "pdf";

const FILTERS: { value: FileFilter; label: string }[] = [
  { value: "all", label: "All" },
  { value: "image", label: "Images" },
  { value: "video", label: "Videos" },
  { value: "pdf", label: "PDFs" },
];

const SEARCH_EXAMPLES = [
  "woman standing with a cat",
  "customer testimonial video",
  "PDF about machine learning",
];

const emptyStats: Stats = {
  total: 0,
  images: 0,
  videos: 0,
  documents: 0,
  pending: 0,
  completed: 0,
  duplicates: 0,
  failed: 0,
  unsupported: 0,
};

function formatFileSize(bytes: number | null): string {
  if (bytes == null) return "unknown size";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatScore(score: number | null): string {
  return typeof score === "number" && Number.isFinite(score) ? score.toFixed(4) : "—";
}

function previewText(asset: Asset): string | null {
  const text = asset.description ?? asset.extracted_text;
  if (!text) return null;
  const singleLine = text.replace(/\s+/g, " ").trim();
  return singleLine.length > PREVIEW_LENGTH
    ? `${singleLine.slice(0, PREVIEW_LENGTH)}…`
    : singleLine;
}

function statusBadgeClass(status: string): string {
  switch (status) {
    case "completed":
      return "bg-emerald-50 text-emerald-700 ring-emerald-600/20";
    case "failed":
      return "bg-red-50 text-red-700 ring-red-600/20";
    case "pending":
    case "processing":
      return "bg-amber-50 text-amber-700 ring-amber-600/25";
    case "duplicate":
      return "bg-violet-50 text-violet-700 ring-violet-600/20";
    default:
      return "bg-slate-100 text-slate-600 ring-slate-500/10";
  }
}

function fileTypeLabel(fileType: string): string {
  if (fileType === "image") return "Image";
  if (fileType === "video") return "Video";
  if (fileType === "pdf") return "PDF";
  return fileType || "File";
}

/* ---------- tiny inline icons (no extra deps) ---------- */

function Icon({ className = "h-4 w-4", children }: { className?: string; children: React.ReactNode }) {
  return (
    <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      {children}
    </svg>
  );
}

const SearchIcon = ({ className = "h-5 w-5" }: { className?: string }) => (
  <Icon className={className}>
    <circle cx="9" cy="9" r="5.5" />
    <path d="m13.5 13.5 3.2 3.2" />
  </Icon>
);

const LayersIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" className="h-5 w-5 text-white" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="m12 2 9 5-9 5-9-5 9-5Z" fill="currentColor" stroke="none" opacity={0.95} />
    <path d="m3 12 9 5 9-5" />
    <path d="m3 17 9 5 9-5" />
  </svg>
);

const ImageIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <rect x="3" y="4" width="14" height="12" rx="2" />
    <circle cx="8.5" cy="9" r="1.4" fill="currentColor" stroke="none" />
    <path d="m4.5 14.5 3.5-3.5 2.5 2.5 2-2 3 3" />
  </Icon>
);

const VideoIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <rect x="2.5" y="5.5" width="11" height="9" rx="2" />
    <path d="m13.5 10 4-2.5v5l-4-2.5Z" fill="currentColor" stroke="none" />
  </Icon>
);

const DocIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <path d="M6 2.5h5.5L15.5 6.5V17.5H6V2.5Z" />
    <path d="M11 2.5v4h4.5" />
    <path d="M8.5 10.5h4M8.5 13h4" />
  </Icon>
);

const BoxIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <path d="M10 2.5 3.5 6v8L10 17.5 16.5 14V6L10 2.5Z" />
    <path d="M3.5 6 10 9.5 16.5 6M10 9.5V17.5" />
  </Icon>
);

const ClockIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <circle cx="10" cy="10" r="7" />
    <path d="M10 6.5V10l2.5 1.5" />
  </Icon>
);

const CheckIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <circle cx="10" cy="10" r="7" />
    <path d="m7.5 10.2 1.8 1.8 3.2-3.7" />
  </Icon>
);

const CopyIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <rect x="7" y="7" width="9" height="9" rx="1.5" />
    <path d="M4.5 11V4.5A1 1 0 0 1 5.5 3.5H11" />
  </Icon>
);

const WarnIcon = ({ className = "h-4 w-4" }: { className?: string }) => (
  <Icon className={className}>
    <path d="M10 3 2.8 15.5h14.4L10 3Z" />
    <path d="M10 8v3.2" />
    <circle cx="10" cy="13.6" r="0.4" fill="currentColor" />
  </Icon>
);

const PlayIcon = () => (
  <svg viewBox="0 0 24 24" className="h-9 w-9 text-white" fill="currentColor" aria-hidden="true">
    <circle cx="12" cy="12" r="11" fill="currentColor" opacity={0.28} />
    <circle cx="12" cy="12" r="11" fill="none" stroke="currentColor" strokeWidth={1.5} opacity={0.6} />
    <path d="M10 8.5v7l6-3.5-6-3.5Z" fill="#fff" />
  </svg>
);

/* ---------- thumbnails ---------- */

function AssetThumbnail({ asset }: { asset: Asset }) {
  const [imgFailed, setImgFailed] = useState(false);
  const fileUrl = `${API_URL}/api/assets/${asset.id}/file`;

  if (asset.file_type === "image" && !imgFailed) {
    return (
      <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-100">
        <img
          src={fileUrl}
          alt=""
          loading="lazy"
          onError={() => setImgFailed(true)}
          className="h-full w-full object-cover"
        />
      </div>
    );
  }

  if (asset.file_type === "video") {
    return (
      <div className="relative flex aspect-[16/10] w-full items-center justify-center overflow-hidden bg-slate-900">
        <div
          className="absolute inset-0 opacity-40"
          style={{
            backgroundImage:
              "radial-gradient(circle at 30% 20%, #334155 0, transparent 45%), radial-gradient(circle at 75% 80%, #1e3a5f 0, transparent 50%)",
          }}
        />
        <PlayIcon />
        {asset.duration_seconds != null && (
          <span className="absolute bottom-2 right-2 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium tabular-nums text-white">
            {asset.duration_seconds.toFixed(1)}s
          </span>
        )}
      </div>
    );
  }

  if (asset.file_type === "pdf") {
    return (
      <div className="relative flex aspect-[16/10] w-full items-center justify-center overflow-hidden bg-slate-100">
        <div className="flex h-20 w-16 flex-col overflow-hidden rounded-md border border-slate-200 bg-white shadow-sm">
          <div className="flex items-center gap-1 border-b border-slate-100 bg-red-50 px-2 py-1.5">
            <span className="h-1.5 w-1.5 rounded-full bg-red-400" />
            <span className="h-1.5 w-1.5 rounded-full bg-amber-300" />
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-300" />
          </div>
          <div className="space-y-1.5 p-2">
            <div className="h-1.5 w-4/5 rounded bg-slate-200" />
            <div className="h-1.5 w-full rounded bg-slate-100" />
            <div className="h-1.5 w-full rounded bg-slate-100" />
            <div className="h-1.5 w-3/5 rounded bg-slate-100" />
          </div>
          <div className="mt-auto flex items-center justify-center gap-1 border-t border-slate-100 py-1 text-[10px] font-semibold tracking-wide text-red-600">
            <DocIcon className="h-3 w-3" /> PDF
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex aspect-[16/10] w-full items-center justify-center bg-slate-100 text-slate-400">
      {asset.file_type === "image" ? <ImageIcon className="h-8 w-8" /> : <BoxIcon className="h-8 w-8" />}
    </div>
  );
}

/* ---------- cards ---------- */

function AssetCard({
  asset,
  score,
  onOpen,
  onKey,
}: {
  asset: Asset;
  score?: number | null;
  onOpen: () => void;
  onKey: (e: React.KeyboardEvent) => void;
}) {
  return (
    <li>
      <article
        onClick={onOpen}
        onKeyDown={onKey}
        role="button"
        tabIndex={0}
        title={`Preview ${asset.filename}`}
        className="group flex h-full cursor-pointer flex-col overflow-hidden rounded-xl border border-slate-200 bg-white shadow-[0_1px_2px_rgba(15,23,42,0.05)] outline-none transition hover:-translate-y-0.5 hover:border-blue-300 hover:shadow-[0_8px_24px_-12px_rgba(37,99,235,0.35)] focus-visible:ring-2 focus-visible:ring-blue-500"
      >
        <AssetThumbnail asset={asset} />
        <div className="flex flex-1 flex-col p-4">
          <div className="flex items-start justify-between gap-2">
            <h3 className="truncate text-sm font-semibold text-slate-900" title={asset.filename}>
              {asset.filename}
            </h3>
            {score !== undefined && (
              <span
                className="shrink-0 rounded-md bg-blue-50 px-2 py-0.5 text-[11px] font-semibold tabular-nums text-blue-700"
                title="Backend similarity score"
              >
                {formatScore(score)}
              </span>
            )}
          </div>
          <p className="mt-0.5 truncate text-xs text-slate-500" title={asset.relative_path}>
            {asset.relative_path}
          </p>
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5 text-[11px] font-medium">
            <span className="inline-flex items-center gap-1 rounded-md bg-slate-100 px-2 py-0.5 text-slate-600">
              {asset.file_type === "image" ? (
                <ImageIcon className="h-3 w-3" />
              ) : asset.file_type === "video" ? (
                <VideoIcon className="h-3 w-3" />
              ) : asset.file_type === "pdf" ? (
                <DocIcon className="h-3 w-3" />
              ) : (
                <BoxIcon className="h-3 w-3" />
              )}
              {fileTypeLabel(asset.file_type)}
            </span>
            <span className={`inline-flex items-center rounded-md px-2 py-0.5 ring-1 ring-inset ${statusBadgeClass(asset.status)}`}>
              {asset.status}
            </span>
            <span className="rounded-md bg-slate-100 px-2 py-0.5 tabular-nums text-slate-600">
              {formatFileSize(asset.file_size)}
            </span>
            {asset.width != null && asset.height != null && (
              <span className="rounded-md bg-slate-100 px-2 py-0.5 tabular-nums text-slate-500">
                {asset.width}×{asset.height}
              </span>
            )}
          </div>
          {previewText(asset) && (
            <p className="mt-2 line-clamp-2 text-xs leading-relaxed text-slate-600">{previewText(asset)}</p>
          )}
          <span className="mt-3 inline-flex items-center gap-1 pt-1 text-xs font-medium text-blue-600 opacity-0 transition group-hover:opacity-100">
            Click to preview <span aria-hidden="true">→</span>
          </span>
        </div>
      </article>
    </li>
  );
}

function StatCard({
  label,
  value,
  icon,
  accent,
}: {
  label: string;
  value: number;
  icon: React.ReactNode;
  accent: string;
}) {
  return (
    <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-3.5 shadow-[0_1px_2px_rgba(15,23,42,0.05)]">
      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${accent}`}>
        {icon}
      </span>
      <span className="min-w-0">
        <span className="block text-xl font-semibold tabular-nums leading-none text-slate-900">{value}</span>
        <span className="mt-1 block truncate text-xs text-slate-500">{label}</span>
      </span>
    </div>
  );
}

export default function App() {
  const [input, setInput] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [fileFilter, setFileFilter] = useState<FileFilter>("all");
  const [backendStatus, setBackendStatus] = useState<string>("checking...");
  const [stats, setStats] = useState<Stats>(emptyStats);
  const [assets, setAssets] = useState<Asset[]>([]);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [searchMethod, setSearchMethod] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [previewId, setPreviewId] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    try {
      const [statsRes, assetsRes] = await Promise.all([
        fetch(`${API_URL}/api/assets/stats`),
        fetch(`${API_URL}/api/assets?limit=50`),
      ]);
      if (statsRes.ok) setStats(await statsRes.json());
      if (assetsRes.ok) {
        const body = await assetsRes.json();
        setAssets(body.items ?? []);
      }
    } catch {
      /* backend may be down; status badge covers it */
    }
  }, []);

  const runSearch = useCallback(
    async (query: string, filter: FileFilter) => {
      const q = query.trim();
      abortRef.current?.abort();
      if (!q) {
        setResults(null);
        setSearchError("");
        setSearchMethod("");
        setSearching(false);
        return;
      }
      const controller = new AbortController();
      abortRef.current = controller;
      setSearching(true);
      setSearchError("");
      try {
        const params = new URLSearchParams({ q, limit: String(SEARCH_LIMIT) });
        if (filter !== "all") params.set("file_type", filter);
        const res = await fetch(`${API_URL}/api/search?${params}`, {
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`Search failed (${res.status})`);
        const body = await res.json();
        setResults(body.results ?? []);
        setSearchMethod(body.method ?? "");
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setResults(null);
        setSearchMethod("");
        setSearchError(err instanceof Error ? err.message : "Search failed");
      } finally {
        if (abortRef.current === controller) {
          abortRef.current = null;
          setSearching(false);
        }
      }
    },
    [],
  );

  // Debounce typing so we don't request on every keystroke.
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(input), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [input]);

  useEffect(() => {
    void runSearch(debouncedQuery, fileFilter);
  }, [debouncedQuery, fileFilter, runSearch]);

  useEffect(() => {
    fetch(`${API_URL}/api/health`)
      .then((res) => res.json())
      .then((data: HealthResponse) => setBackendStatus(data.status))
      .catch(() => setBackendStatus("unreachable"));
    void refresh();
    return () => abortRef.current?.abort();
  }, [refresh]);

  async function runIndexing() {
    setBusy(true);
    setMessage("");
    try {
      const indexRes = await fetch(`${API_URL}/api/index`, { method: "POST" });
      if (!indexRes.ok) throw new Error(`Index failed (${indexRes.status})`);
      const indexBody = await indexRes.json();
      const processRes = await fetch(`${API_URL}/api/process`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ limit: 200 }),
      });
      if (!processRes.ok) throw new Error(`Process failed (${processRes.status})`);
      const processBody = await processRes.json();
      setMessage(
        `Indexed: +${indexBody.added} ~${indexBody.updated} dup ${indexBody.duplicates}. ` +
          `AI: ${processBody.completed}/${processBody.processed} completed.`,
      );
      await refresh();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Indexing failed");
    } finally {
      setBusy(false);
    }
  }

  function submitSearch() {
    // Flush the debounce and search immediately.
    setDebouncedQuery(input);
    void runSearch(input, fileFilter);
  }

  function clearSearch() {
    abortRef.current?.abort();
    setInput("");
    setDebouncedQuery("");
    setResults(null);
    setSearchError("");
    setSearchMethod("");
  }

  const hasQuery = debouncedQuery.trim().length > 0;
  const notIndexed = stats.total === 0;

  function cardKeyHandler(e: React.KeyboardEvent, id: string) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      setPreviewId(id);
    }
  }

  const backendOnline = backendStatus === "ok" || backendStatus === "healthy";
  const backendDown = backendStatus === "unreachable" || backendStatus === "error";
  const statusDot = backendOnline ? "bg-emerald-500" : backendDown ? "bg-red-500" : "bg-amber-500";
  const statusText = backendOnline ? "Online" : backendDown ? "Unreachable" : backendStatus;
  const showingResults = results !== null || searchError !== "";

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 antialiased">
      {/* ---------- Top navigation ---------- */}
      <header className="sticky top-0 z-40 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex h-16 w-full max-w-7xl items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-600 shadow-sm">
              <LayersIcon />
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[15px] font-semibold leading-tight tracking-tight">
                AI Digital Asset Manager
              </span>
              <span className="block text-xs text-slate-500">Local-first media library · AI search</span>
            </span>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span
              className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs font-medium text-slate-600 shadow-sm"
              title={`Backend status: ${backendStatus}`}
            >
              <span className="relative flex h-2 w-2">
                <span className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-60 ${statusDot}`} />
                <span className={`relative inline-flex h-2 w-2 rounded-full ${statusDot}`} />
              </span>
              Backend · {statusText}
            </span>
            <span className="hidden items-center gap-1.5 rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-500 md:inline-flex">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
              Local AI · no downloads
            </span>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-7xl space-y-5 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        {/* ---------- Search hero ---------- */}
        <section className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_1px_3px_rgba(15,23,42,0.06)]">
          <div className="border-b border-slate-100 bg-slate-50/60 px-5 pb-4 pt-5 sm:px-7">
            <div className="flex flex-wrap items-center gap-2">
              <span className="inline-flex items-center gap-1.5 rounded-full bg-blue-600 px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wide text-white">
                <SearchIcon className="h-3.5 w-3.5" /> AI Search
              </span>
              <h2 className="text-lg font-semibold tracking-tight">Search your assets</h2>
            </div>
            <p className="mt-1.5 max-w-2xl text-sm text-slate-500">
              Describe what you're looking for in plain language. Search covers AI descriptions
              and extracted text, ranked by similarity.
            </p>
          </div>

          <div className="px-5 py-5 sm:px-7 sm:py-6">
            <div className="flex flex-col gap-3 lg:flex-row">
              <label className="relative block flex-1">
                <span className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-slate-400">
                  <SearchIcon />
                </span>
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") submitSearch();
                  }}
                  placeholder='Try "customer testimonial video" or "PDF about machine learning"…'
                  aria-label="Search assets"
                  className="w-full rounded-xl border border-slate-300 bg-white py-3.5 pl-11 pr-11 text-[15px] shadow-sm outline-none transition placeholder:text-slate-400 focus:border-blue-500 focus:ring-4 focus:ring-blue-500/10"
                />
                {input && (
                  <button
                    type="button"
                    onClick={clearSearch}
                    aria-label="Clear search input"
                    className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                  >
                    ✕
                  </button>
                )}
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={submitSearch}
                  disabled={searching}
                  className="inline-flex flex-1 items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 py-3.5 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-500/30 disabled:cursor-not-allowed disabled:opacity-60 lg:flex-none"
                >
                  {searching ? (
                    <>
                      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden="true" />
                      Searching…
                    </>
                  ) : (
                    <>
                      <SearchIcon className="h-4 w-4" /> Search
                    </>
                  )}
                </button>
                {(hasQuery || input) && (
                  <button
                    type="button"
                    onClick={clearSearch}
                    className="rounded-xl border border-slate-300 bg-white px-4 py-3.5 text-sm font-medium text-slate-600 shadow-sm hover:bg-slate-50"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>

            <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium text-slate-500">Try:</span>
                {SEARCH_EXAMPLES.map((example) => (
                  <button
                    key={example}
                    type="button"
                    onClick={() => setInput(example)}
                    className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-600 transition hover:border-blue-300 hover:bg-blue-50 hover:text-blue-700"
                  >
                    “{example}”
                  </button>
                ))}
              </div>
              <div className="flex flex-wrap gap-1.5 rounded-xl bg-slate-100 p-1 sm:shrink-0" role="group" aria-label="File type filter">
                {FILTERS.map((f) => (
                  <button
                    key={f.value}
                    type="button"
                    onClick={() => setFileFilter(f.value)}
                    aria-pressed={fileFilter === f.value}
                    className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                      fileFilter === f.value
                        ? "bg-white text-blue-700 shadow-sm ring-1 ring-slate-200"
                        : "text-slate-500 hover:text-slate-800"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {searchMethod && (
              <p className="mt-3 border-t border-dashed border-slate-200 pt-3 text-xs text-slate-500">
                Search method: <span className="font-medium text-slate-600">{searchMethod}</span>
              </p>
            )}
          </div>
        </section>

        {/* ---------- Results ---------- */}
        {showingResults && (
          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)] sm:p-7" aria-live="polite">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-base font-semibold tracking-tight">Search results</h2>
              {results !== null && !searching && !searchError && (
                <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
                  {results.length} result{results.length === 1 ? "" : "s"}
                  {hasQuery && <> for “{debouncedQuery.trim()}”</>}
                </span>
              )}
            </div>

            {searching && (
              <div className="mt-6 flex items-center gap-3 text-sm text-slate-500">
                <span className="inline-block h-5 w-5 animate-spin rounded-full border-2 border-slate-200 border-t-blue-600" aria-hidden="true" />
                Searching your library…
              </div>
            )}

            {!searching && searchError && (
              <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-4 text-sm">
                <p className="font-semibold text-red-800">Search failed</p>
                <p className="mt-1 text-red-700">{searchError}</p>
                <button
                  type="button"
                  onClick={submitSearch}
                  className="mt-3 rounded-lg bg-red-600 px-3.5 py-1.5 text-xs font-semibold text-white hover:bg-red-700"
                >
                  Retry search
                </button>
              </div>
            )}

            {!searching && !searchError && hasQuery && results !== null && results.length === 0 && (
              <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-10 text-center">
                <span className="mx-auto flex h-11 w-11 items-center justify-center rounded-full bg-white text-slate-400 shadow-sm ring-1 ring-slate-200">
                  <SearchIcon />
                </span>
                <p className="mt-3 text-sm font-semibold text-slate-800">No results for “{debouncedQuery.trim()}”</p>
                <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
                  {notIndexed
                    ? "Your library isn't indexed yet — run Start Indexing below, then try again."
                    : "Try different words, switch the file-type filter, or run indexing to pick up new files."}
                </p>
              </div>
            )}

            {!searching && !searchError && results !== null && results.length > 0 && (
              <ul className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {results.map(({ asset, score }) => (
                  <AssetCard
                    key={asset.id}
                    asset={asset}
                    score={score}
                    onOpen={() => setPreviewId(asset.id)}
                    onKey={(e) => cardKeyHandler(e, asset.id)}
                  />
                ))}
              </ul>
            )}
          </section>
        )}

        {/* ---------- Indexing + stats ---------- */}
        <div className="grid gap-5 lg:grid-cols-[340px_1fr]">
          <section className="flex flex-col rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)] sm:p-6">
            <h2 className="text-base font-semibold tracking-tight">Dataset</h2>
            <p className="mt-1 text-sm text-slate-500">
              Dataset path:{" "}
              <code className="rounded-md bg-slate-100 px-1.5 py-0.5 font-mono text-xs text-slate-700">./data</code>
            </p>

            <button
              type="button"
              onClick={runIndexing}
              disabled={busy}
              className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-blue-700 focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-blue-500/30 disabled:cursor-not-allowed disabled:opacity-60"
            >
              {busy ? (
                <>
                  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/40 border-t-white" aria-hidden="true" />
                  Indexing + AI processing…
                </>
              ) : (
                <>▶ Start Indexing</>
              )}
            </button>

            {busy && (
              <div className="mt-3" aria-hidden="true">
                <div className="h-1.5 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full w-1/2 animate-pulse rounded-full bg-blue-500" />
                </div>
                <p className="mt-2 text-xs text-slate-500">Scanning files and generating AI descriptions…</p>
              </div>
            )}

            {message && (
              <p className="mt-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs leading-relaxed text-slate-600">
                {message}
              </p>
            )}
            <p className="mt-3 text-xs leading-relaxed text-slate-500">
              Scans files, then runs local AI (descriptions + embeddings). No model downloads.
            </p>

            <button
              type="button"
              onClick={() => void refresh()}
              className="mt-auto inline-flex items-center justify-center gap-1.5 rounded-lg border border-slate-200 px-3 py-2 pt-2 text-xs font-medium text-slate-600 hover:bg-slate-50"
              style={{ marginTop: "1rem" }}
            >
              ↻ Refresh status
            </button>
          </section>

          <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)] sm:p-6">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-semibold tracking-tight">Indexing statistics</h2>
              <span className="text-xs text-slate-400">{stats.total} assets tracked</span>
            </div>
            <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-4">
              <StatCard label="Total assets" value={stats.total} accent="bg-blue-50 text-blue-600" icon={<BoxIcon />} />
              <StatCard label="Images" value={stats.images} accent="bg-emerald-50 text-emerald-600" icon={<ImageIcon />} />
              <StatCard label="Videos" value={stats.videos} accent="bg-violet-50 text-violet-600" icon={<VideoIcon />} />
              <StatCard label="PDF documents" value={stats.documents} accent="bg-red-50 text-red-600" icon={<DocIcon />} />
              <StatCard label="Pending" value={stats.pending} accent="bg-amber-50 text-amber-600" icon={<ClockIcon />} />
              <StatCard label="Completed" value={stats.completed} accent="bg-emerald-50 text-emerald-600" icon={<CheckIcon />} />
              <StatCard label="Duplicates" value={stats.duplicates} accent="bg-violet-50 text-violet-600" icon={<CopyIcon />} />
              <StatCard label="Failed" value={stats.failed} accent="bg-red-50 text-red-600" icon={<WarnIcon />} />
            </div>
          </section>
        </div>

        {/* ---------- Recent assets ---------- */}
        <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_1px_3px_rgba(15,23,42,0.06)] sm:p-7">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold tracking-tight">Recent assets</h2>
              <p className="mt-0.5 text-xs text-slate-500">
                {assets.length > 0 ? "Click any asset to open the full preview." : "Your indexed files will appear here."}
              </p>
            </div>
            {assets.length > 0 && (
              <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-medium text-slate-600">
                Showing {Math.min(assets.length, 20)} of {assets.length}
              </span>
            )}
          </div>

          {assets.length === 0 ? (
            <div className="mt-5 rounded-xl border border-dashed border-slate-300 bg-slate-50 px-6 py-12 text-center sm:py-14">
              <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-white text-slate-400 shadow-sm ring-1 ring-slate-200">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} className="h-7 w-7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <rect x="3" y="3" width="18" height="18" rx="3" />
                  <circle cx="9" cy="9" r="1.6" />
                  <path d="m5 19 5.5-5.5 2.5 2.5 2-2L19.5 18" />
                </svg>
              </span>
              <h3 className="mt-4 text-sm font-semibold text-slate-900">No assets indexed yet</h3>
              <p className="mx-auto mt-1 max-w-md text-sm text-slate-500">
                Add image, video, or PDF files to the{" "}
                <code className="rounded bg-white px-1 font-mono text-xs ring-1 ring-slate-200">data/</code>{" "}
                folder, then run indexing to generate AI descriptions and make everything searchable.
              </p>
              <button
                type="button"
                onClick={runIndexing}
                disabled={busy}
                className="mt-5 inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-700 disabled:opacity-60"
              >
                {busy ? "Indexing…" : "▶ Start Indexing"}
              </button>
              <p className="mt-3 text-xs text-slate-400">Takes a few seconds for small datasets</p>
            </div>
          ) : (
            <ul className="mt-5 grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {assets.slice(0, 20).map((a) => (
                <AssetCard
                  key={a.id}
                  asset={a}
                  onOpen={() => setPreviewId(a.id)}
                  onKey={(e) => cardKeyHandler(e, a.id)}
                />
              ))}
            </ul>
          )}
        </section>

        <footer className="flex flex-wrap items-center justify-between gap-2 px-1 pb-4 text-xs text-slate-400">
          <span>AI Digital Asset Manager · local-first · React + Tailwind</span>
          <span>
            Backend: {API_URL} · {statusText}
          </span>
        </footer>
      </main>

      {previewId && (
        <PreviewModal assetId={previewId} apiUrl={API_URL} onClose={() => setPreviewId(null)} />
      )}
    </div>
  );
}
