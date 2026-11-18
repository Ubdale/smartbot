// Mirrors the FastAPI response schemas (app/schemas/*.py).

export type User = {
  id: string;
  email: string;
  role: string;
  workspace_id: string;
  created_at: string;
};

export type ColumnKind = "number" | "text" | "date" | "datetime" | "boolean";

export type Column = {
  name: string;
  type: string;
  kind: ColumnKind;
  null_count: number;
  distinct_count: number;
  samples: unknown[];
  min: unknown;
  max: unknown;
};

export type DatasetSummary = {
  id: string;
  name: string;
  original_filename: string;
  file_format: "csv" | "xlsx";
  sheet_name: string | null;
  size_bytes: number;
  row_count: number;
  version: number;
  created_at: string;
};

export type Dataset = DatasetSummary & { columns: Column[] };

export type UploadResult = Dataset & {
  sheet_names: string[];
  detected_encoding: string | null;
};

export type Preview = {
  dataset_id: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
};

export type Chart = {
  type: "bar" | "line" | "pie";
  x: string;
  y: string;
  series: string | null;
};

export type Answer = {
  table: { columns: string[]; rows: unknown[][]; truncated: boolean };
  chart: Chart | null;
  explanation: {
    description: string;
    sql: string;
    params: unknown[];
    date_range: { start: string; end: string; label: string } | null;
    plan: Record<string, unknown>;
    cached: boolean;
    elapsed_ms: number;
  };
  summary_source: "llm" | "template";
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "answered" | "clarification" | "failed" | null;
  answer: Answer | null;
  created_at: string;
};

export type ConversationSummary = {
  id: string;
  title: string;
  dataset_id: string;
  created_at: string;
  updated_at: string;
};

export type Conversation = ConversationSummary & { messages: Message[] };

export type AskResponse = {
  conversation_id: string;
  question: Message;
  reply: Message;
};
