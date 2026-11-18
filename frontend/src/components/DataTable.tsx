import { formatValue, prettyColumn } from "@/lib/format";

export function DataTable({
  columns,
  rows,
  truncated = false,
  pretty = true,
  maxHeight = "max-h-96",
}: {
  columns: string[];
  rows: unknown[][];
  truncated?: boolean;
  pretty?: boolean;
  maxHeight?: string;
}) {
  const numeric = columns.map((_, i) => rows.some((r) => typeof r[i] === "number"));
  return (
    <div>
      <div className={`overflow-auto rounded-lg border border-zinc-200 dark:border-zinc-800 ${maxHeight}`}>
        <table className="w-full border-collapse text-sm">
          <thead className="sticky top-0 bg-zinc-50 dark:bg-zinc-900">
            <tr>
              {columns.map((c, i) => (
                <th
                  key={c}
                  scope="col"
                  className={`whitespace-nowrap border-b border-zinc-200 px-3 py-2 font-medium text-zinc-600 dark:border-zinc-800 dark:text-zinc-300 ${
                    numeric[i] ? "text-right" : "text-left"
                  }`}
                >
                  {pretty ? prettyColumn(c) : c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={r} className="odd:bg-white even:bg-zinc-50/60 dark:odd:bg-zinc-950 dark:even:bg-zinc-900/50">
                {row.map((v, i) => (
                  <td
                    key={i}
                    className={`whitespace-nowrap px-3 py-1.5 ${
                      numeric[i] ? "text-right tabular-nums" : ""
                    } ${v === null ? "text-zinc-400" : ""}`}
                  >
                    {formatValue(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-1.5 text-xs text-zinc-500">
        {rows.length} {rows.length === 1 ? "row" : "rows"}
        {truncated && " (more exist; ask for a higher limit or add a filter)"}
      </p>
    </div>
  );
}
