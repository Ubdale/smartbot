import type { Column } from "@/lib/types";

/** A few starter questions built from the dataset's own columns. */
export function suggestQuestions(columns: Column[]): string[] {
  const text = columns.filter((c) => c.kind === "text" && c.distinct_count > 1 && c.distinct_count <= 50);
  const nums = columns.filter((c) => c.kind === "number");
  const dates = columns.filter((c) => c.kind === "date" || c.kind === "datetime");
  const human = (c: Column) => c.name.replaceAll("_", " ");
  const out: string[] = [];

  if (nums[0] && text[0]) out.push(`Top 10 ${human(text[0])} by total ${human(nums.at(-1)!)}`);
  if (nums[0] && dates[0]) out.push(`Monthly ${human(nums.at(-1)!)} trend this year`);
  if (dates[0]) out.push(`How many rows last quarter?`);
  if (text[1] && nums[0]) out.push(`Average ${human(nums[0])} by ${human(text[1])}`);
  if (text[0] && out.length < 4) out.push(`Count by ${human(text[0])}`);
  return out.slice(0, 4);
}
