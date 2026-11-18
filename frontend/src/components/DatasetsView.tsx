"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type DragEvent } from "react";

import { Button, Card, ErrorNote, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatBytes, formatDate } from "@/lib/format";
import type { DatasetSummary, UploadResult } from "@/lib/types";

const MAX_MB = 20;

export function DatasetsView() {
  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [uploading, setUploading] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api<DatasetSummary[]>("/datasets").then(setDatasets).catch((e: ApiError) => setError(e.message));
  }, []);

  async function upload(file: File) {
    setError(null);
    setNotice(null);
    if (!/\.(csv|xlsx)$/i.test(file.name)) return setError("Only .csv and .xlsx files are supported.");
    if (file.size > MAX_MB * 1024 * 1024) return setError(`That file is over the ${MAX_MB} MB limit.`);
    setUploading(file.name);
    try {
      const form = new FormData();
      form.append("file", file);
      const ds = await api<UploadResult>("/datasets", { form });
      setDatasets((list) => [ds, ...(list ?? [])]);
      const others = ds.sheet_names.filter((s) => s !== ds.sheet_name);
      setNotice(
        `Added “${ds.name}”: ${ds.row_count.toLocaleString()} rows, ${ds.columns.length} columns.` +
          (others.length ? ` Used sheet “${ds.sheet_name}” (other sheets: ${others.join(", ")}).` : ""),
      );
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Upload failed.");
    } finally {
      setUploading(null);
      if (fileInput.current) fileInput.current.value = "";
    }
  }

  async function remove(ds: DatasetSummary) {
    if (!confirm(`Delete “${ds.name}” and its conversations?`)) return;
    try {
      await api(`/datasets/${ds.id}`, { method: "DELETE" });
      setDatasets((list) => list?.filter((d) => d.id !== ds.id) ?? null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Delete failed.");
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) void upload(file);
  }

  return (
    <main className="mx-auto w-full max-w-5xl space-y-6 px-4 py-8">
      <div>
        <h1 className="text-2xl font-semibold">Datasets</h1>
        <p className="mt-1 text-sm text-zinc-500">
          Upload a CSV or Excel file. Only column names, types and a few sample values are shared
          with the AI, never your rows.
        </p>
      </div>

      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`flex flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
          dragging
            ? "border-indigo-500 bg-indigo-50 dark:bg-indigo-950/30"
            : "border-zinc-300 bg-white dark:border-zinc-700 dark:bg-zinc-900"
        }`}
      >
        {uploading ? (
          <p className="flex items-center gap-2 text-sm">
            <Spinner /> Uploading and analysing {uploading}…
          </p>
        ) : (
          <>
            <p className="text-sm font-medium">Drag a file here, or</p>
            <Button className="mt-3" onClick={() => fileInput.current?.click()}>
              Choose a file
            </Button>
            <p className="mt-2 text-xs text-zinc-500">.csv or .xlsx, up to {MAX_MB} MB</p>
          </>
        )}
        <input
          ref={fileInput}
          type="file"
          accept=".csv,.xlsx"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
      </div>

      {notice && (
        <p className="rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-200">
          {notice}
        </p>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}

      {datasets === null ? (
        <p className="flex items-center gap-2 text-sm text-zinc-500">
          <Spinner /> Loading…
        </p>
      ) : datasets.length === 0 ? (
        <p className="text-sm text-zinc-500">No datasets yet.</p>
      ) : (
        <Card>
          <ul className="divide-y divide-zinc-200 dark:divide-zinc-800">
            {datasets.map((ds) => (
              <li key={ds.id} className="flex flex-wrap items-center gap-3 px-4 py-3">
                <div className="min-w-0 flex-1">
                  <Link href={`/datasets/${ds.id}`} className="font-medium hover:underline">
                    {ds.name}
                  </Link>
                  <p className="text-xs text-zinc-500">
                    {ds.row_count.toLocaleString()} rows · {ds.file_format.toUpperCase()} ·{" "}
                    {formatBytes(ds.size_bytes)} · added {formatDate(ds.created_at)}
                  </p>
                </div>
                <Link href={`/chat?d=${ds.id}`}>
                  <Button variant="ghost">Ask a question</Button>
                </Link>
                <Button variant="danger" onClick={() => remove(ds)}>
                  Delete
                </Button>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </main>
  );
}
