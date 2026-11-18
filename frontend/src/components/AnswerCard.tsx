import { Chart } from "@/components/Chart";
import { DataTable } from "@/components/DataTable";
import { formatValue } from "@/lib/format";
import type { Message } from "@/lib/types";

export function AnswerCard({ message }: { message: Message }) {
  if (message.status === "clarification") {
    return (
      <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm dark:border-amber-900 dark:bg-amber-950/30">
        <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-amber-700 dark:text-amber-300">
          Need a bit more detail
        </p>
        {message.content}
      </div>
    );
  }
  if (message.status === "failed" || !message.answer) {
    return (
      <div className="whitespace-pre-line rounded-xl border border-zinc-200 bg-white px-4 py-3 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300">
        {message.content}
      </div>
    );
  }

  const { table, chart, explanation, summary_source } = message.answer;
  return (
    <div className="space-y-4 rounded-xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <p className="text-[15px] leading-relaxed">{message.content}</p>

      {chart && <Chart spec={chart} columns={table.columns} rows={table.rows} />}

      {table.rows.length > 0 && (
        <DataTable columns={table.columns} rows={table.rows} truncated={table.truncated} />
      )}

      <details className="group rounded-lg border border-zinc-200 text-sm dark:border-zinc-800">
        <summary className="cursor-pointer select-none px-3 py-2 font-medium text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white">
          How this was calculated
        </summary>
        <div className="space-y-3 border-t border-zinc-200 px-3 py-3 dark:border-zinc-800">
          <p>{explanation.description}</p>
          {explanation.date_range && (
            <p className="text-zinc-600 dark:text-zinc-400">
              &ldquo;{explanation.date_range.label}&rdquo; means {explanation.date_range.start} to{" "}
              {explanation.date_range.end} (inclusive).
            </p>
          )}
          <div>
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
              Query (values passed separately as parameters)
            </p>
            <pre className="overflow-x-auto rounded-md bg-zinc-100 p-3 font-mono text-xs leading-relaxed dark:bg-zinc-950">
              {explanation.sql}
            </pre>
            {explanation.params.length > 0 && (
              <p className="mt-1 font-mono text-xs text-zinc-500">
                Parameters: {explanation.params.map((p) => formatValue(p)).join(", ")}
              </p>
            )}
          </div>
          <p className="text-xs text-zinc-500">
            Every number above was computed by the database engine, not the AI.{" "}
            {summary_source === "llm"
              ? "The AI wrote the summary from these results, and each number in it was checked against them."
              : "The summary was generated from the results."}
            {explanation.cached ? " Served from cache." : ` Ran in ${explanation.elapsed_ms} ms.`}
          </p>
        </div>
      </details>
    </div>
  );
}
