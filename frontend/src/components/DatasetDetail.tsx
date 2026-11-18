"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { DataTable } from "@/components/DataTable";
import { Button, Card, ErrorNote, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatValue } from "@/lib/format";
import type { Dataset, Preview } from "@/lib/types";

const KIND_STYLE: Record<string, string> = {
  number: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  text: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  date: "bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-300",
  datetime: "bg-violet-100 text-violet-800 dark:bg-violet-950 dark:text-violet-300",
  boolean: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
};

export function DatasetDetail({ id }: { id: string }) {
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Dataset>(`/datasets/${id}`).then(setDataset).catch((e: ApiError) => setError(e.message));
    api<Preview>(`/datasets/${id}/preview?limit=50`).then(setPreview).catch(() => undefined);
  }, [id]);

  if (error) {
    return (
      <main className="mx-auto w-full max-w-5xl px-4 py-8">
        <ErrorNote>{error}</ErrorNote>
      </main>
    );
  }
  if (!dataset) {
    return (
      <main className="mx-auto flex w-full max-w-5xl items-center gap-2 px-4 py-8 text-sm text-zinc-500">
        <Spinner /> Loading…
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-6xl space-y-6 px-4 py-8">
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex-1">
          <Link href="/datasets" className="text-sm text-zinc-500 hover:underline">
            ← Datasets
          </Link>
          <h1 className="mt-1 text-2xl font-semibold">{dataset.name}</h1>
          <p className="text-sm text-zinc-500">
            {dataset.row_count.toLocaleString()} rows · {dataset.columns.length} columns · from{" "}
            {dataset.original_filename}
            {dataset.sheet_name && ` (sheet “${dataset.sheet_name}”)`}
          </p>
        </div>
        <Link href={`/chat?d=${dataset.id}`}>
          <Button>Ask a question</Button>
        </Link>
      </div>

      <Card className="overflow-hidden">
        <h2 className="border-b border-zinc-200 px-4 py-3 text-sm font-semibold dark:border-zinc-800">
          Columns
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-zinc-500">
              <tr>
                <th className="px-4 py-2 font-medium">Name</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">Examples</th>
                <th className="px-4 py-2 font-medium">Range</th>
                <th className="px-4 py-2 text-right font-medium">Empty</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-100 dark:divide-zinc-800">
              {dataset.columns.map((c) => (
                <tr key={c.name}>
                  <td className="px-4 py-2 font-medium">{c.name}</td>
                  <td className="px-4 py-2">
                    <span className={`rounded-md px-1.5 py-0.5 text-xs font-medium ${KIND_STYLE[c.kind]}`}>
                      {c.kind}
                    </span>
                  </td>
                  <td className="max-w-xs truncate px-4 py-2 text-zinc-600 dark:text-zinc-400">
                    {c.samples.map((s) => formatValue(s)).join(", ")}
                  </td>
                  <td className="whitespace-nowrap px-4 py-2 text-zinc-600 dark:text-zinc-400">
                    {c.min !== null && c.min !== undefined
                      ? `${formatValue(c.min)} – ${formatValue(c.max)}`
                      : `${c.distinct_count.toLocaleString()} distinct`}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums text-zinc-600 dark:text-zinc-400">
                    {c.null_count.toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <section>
        <h2 className="mb-2 text-sm font-semibold">First {preview?.rows.length ?? 50} rows</h2>
        {preview ? (
          <DataTable columns={preview.columns} rows={preview.rows} pretty={false} maxHeight="max-h-[32rem]" />
        ) : (
          <p className="flex items-center gap-2 text-sm text-zinc-500">
            <Spinner /> Loading preview…
          </p>
        )}
      </section>
    </main>
  );
}
