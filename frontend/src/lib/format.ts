const numberFmt = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 });

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return numberFmt.format(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return String(value);
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** Column headers like "sum_revenue" -> "Total revenue". */
export function prettyColumn(name: string): string {
  const m = /^(count_distinct|count|sum|avg|min|max)_(.+)$/.exec(name);
  if (name === "count") return "Count";
  if (!m) return name;
  const label = { count: "Count of", count_distinct: "Distinct", sum: "Total", avg: "Average", min: "Minimum", max: "Maximum" }[m[1]];
  return `${label} ${m[2].replaceAll("_", " ")}`;
}
