"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";

import { AnswerCard } from "@/components/AnswerCard";
import { Button, ErrorNote, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { suggestQuestions } from "@/lib/suggestions";
import type {
  AskResponse,
  Conversation,
  ConversationSummary,
  Dataset,
  DatasetSummary,
  Message,
} from "@/lib/types";

function useElapsedSeconds(running: boolean): number {
  const [seconds, setSeconds] = useState(0);
  useEffect(() => {
    if (!running) return;
    const started = Date.now();
    const id = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => {
      clearInterval(id);
      setSeconds(0);
    };
  }, [running]);
  return seconds;
}

export function ChatView() {
  const router = useRouter();
  const params = useSearchParams();
  const conversationId = params.get("c");
  const datasetParam = params.get("d");

  const [datasets, setDatasets] = useState<DatasetSummary[] | null>(null);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  // The loaded thread, tagged with its conversation id (null = a new, unsaved chat).
  const [thread, setThread] = useState<{ id: string | null; datasetId: string | null; messages: Message[] }>({
    id: null,
    datasetId: null,
    messages: [],
  });
  const [chosenDatasetId, setChosenDatasetId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Dataset | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const elapsed = useElapsedSeconds(pending !== null);

  // Everything below is derived from the URL + loaded data (no state syncing in effects).
  const threadLoaded = thread.id === conversationId;
  const messages = threadLoaded ? thread.messages : [];
  const datasetId = conversationId
    ? threadLoaded
      ? thread.datasetId
      : null
    : (datasets?.find((d) => d.id === (chosenDatasetId ?? datasetParam))?.id ?? datasets?.[0]?.id ?? null);
  const dataset = detail && detail.id === datasetId ? detail : null;

  // Datasets + conversation list.
  useEffect(() => {
    api<DatasetSummary[]>("/datasets").then(setDatasets).catch((e: ApiError) => setError(e.message));
    api<ConversationSummary[]>("/chat/conversations").then(setConversations).catch(() => undefined);
  }, []);

  // Load the conversation named in the URL, unless it's already the loaded thread.
  useEffect(() => {
    if (!conversationId || threadLoaded) return;
    api<Conversation>(`/chat/conversations/${conversationId}`)
      .then((conv) => setThread({ id: conv.id, datasetId: conv.dataset_id, messages: conv.messages }))
      .catch((e: ApiError) => setError(e.message));
  }, [conversationId, threadLoaded]);

  // Column details for the selected dataset (starter questions + header).
  useEffect(() => {
    if (!datasetId) return;
    api<Dataset>(`/datasets/${datasetId}`).then(setDetail).catch(() => undefined);
  }, [datasetId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, pending]);

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || pending) return;
      setPending(q);
      setError(null);
      setInput("");
      try {
        const res = await api<AskResponse>("/chat/ask", {
          json: conversationId
            ? { question: q, conversation_id: conversationId }
            : { question: q, dataset_id: datasetId },
        });
        setThread((t) => ({
          id: res.conversation_id,
          datasetId,
          messages: [...(t.id === res.conversation_id ? t.messages : []), res.question, res.reply],
        }));
        if (!conversationId) router.replace(`/chat?c=${res.conversation_id}`, { scroll: false });
        api<ConversationSummary[]>("/chat/conversations").then(setConversations).catch(() => undefined);
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Something went wrong.");
        setInput(q);
      } finally {
        setPending(null);
      }
    },
    [conversationId, datasetId, pending, router],
  );

  async function removeConversation(id: string) {
    if (!confirm("Delete this conversation?")) return;
    await api(`/chat/conversations/${id}`, { method: "DELETE" }).catch(() => undefined);
    setConversations((cs) => cs.filter((c) => c.id !== id));
    if (id === conversationId) router.replace("/chat");
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    void ask(input);
  }

  if (datasets && datasets.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-8 text-center">
        <div className="max-w-sm space-y-3">
          <h2 className="text-lg font-semibold">Upload some data first</h2>
          <p className="text-sm text-zinc-500">
            Add a CSV or Excel file, then ask questions like &ldquo;top 10 customers by
            revenue&rdquo;.
          </p>
          <Link href="/datasets">
            <Button>Go to datasets</Button>
          </Link>
        </div>
      </div>
    );
  }

  const isNew = !conversationId;
  const suggestions = dataset ? suggestQuestions(dataset.columns) : [];

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-1">
      {/* Conversations */}
      <aside className="hidden w-64 shrink-0 flex-col border-r border-zinc-200 p-3 md:flex dark:border-zinc-800">
        <Button variant="ghost" className="mb-2 justify-start" onClick={() => router.push("/chat")}>
          + New question
        </Button>
        <div className="space-y-0.5 overflow-y-auto">
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`group flex items-center rounded-lg text-sm ${
                c.id === conversationId
                  ? "bg-zinc-200/70 dark:bg-zinc-800"
                  : "hover:bg-zinc-100 dark:hover:bg-zinc-900"
              }`}
            >
              <Link href={`/chat?c=${c.id}`} className="flex-1 truncate px-3 py-2" title={c.title}>
                {c.title}
              </Link>
              <button
                onClick={() => removeConversation(c.id)}
                className="mr-1 hidden rounded px-1.5 text-zinc-400 hover:text-red-600 group-hover:block"
                aria-label="Delete conversation"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* Thread */}
      <section className="flex min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-zinc-200 px-4 py-2.5 text-sm dark:border-zinc-800">
          <span className="text-zinc-500">Dataset:</span>
          {isNew && datasets ? (
            <select
              value={datasetId ?? ""}
              onChange={(e) => setChosenDatasetId(e.target.value)}
              className="rounded-md border border-zinc-300 bg-white px-2 py-1 dark:border-zinc-700 dark:bg-zinc-900"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count.toLocaleString()} rows)
                </option>
              ))}
            </select>
          ) : (
            <span className="font-medium">{dataset?.name ?? "…"}</span>
          )}
          <button
            className="ml-auto rounded-md px-2 py-1 text-zinc-600 hover:bg-zinc-100 md:hidden dark:text-zinc-400 dark:hover:bg-zinc-800"
            onClick={() => router.push("/chat")}
          >
            New
          </button>
        </div>

        <div className="flex-1 space-y-5 overflow-y-auto px-4 py-6">
          {isNew && messages.length === 0 && !pending && (
            <div className="mx-auto max-w-2xl pt-6 text-center">
              <h1 className="text-2xl font-semibold">What would you like to know?</h1>
              <p className="mt-2 text-sm text-zinc-500">
                Ask in plain English. The AI turns your question into a query, and the numbers are
                computed directly from your data.
              </p>
              {suggestions.length > 0 && (
                <div className="mt-6 grid gap-2 sm:grid-cols-2">
                  {suggestions.map((s) => (
                    <button
                      key={s}
                      onClick={() => ask(s)}
                      className="rounded-xl border border-zinc-200 bg-white px-4 py-3 text-left text-sm hover:border-indigo-300 hover:bg-indigo-50/50 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-indigo-800 dark:hover:bg-indigo-950/30"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {messages.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="flex justify-end">
                <p className="max-w-[80%] rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2 text-[15px] text-white">
                  {m.content}
                </p>
              </div>
            ) : (
              <div key={m.id} className="max-w-4xl">
                <AnswerCard message={m} />
              </div>
            ),
          )}

          {pending && (
            <>
              <div className="flex justify-end">
                <p className="max-w-[80%] rounded-2xl rounded-br-md bg-indigo-600/70 px-4 py-2 text-[15px] text-white">
                  {pending}
                </p>
              </div>
              <div className="flex items-center gap-2 text-sm text-zinc-500">
                <Spinner />
                Working it out… {elapsed > 0 && `${elapsed}s`}
                {elapsed >= 15 && (
                  <span className="text-xs">
                    (a local AI model on a CPU can take a minute or more)
                  </span>
                )}
              </div>
            </>
          )}
          {error && (
            <div className="max-w-2xl">
              <ErrorNote>{error}</ErrorNote>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        <form
          onSubmit={onSubmit}
          className="sticky bottom-0 border-t border-zinc-200 bg-[var(--background)] px-4 py-3 dark:border-zinc-800"
        >
          <div className="mx-auto flex max-w-4xl gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              maxLength={500}
              placeholder={
                isNew ? "e.g. Top 10 customers by revenue last quarter" : "Ask a follow-up, e.g. “now split by city”"
              }
              className="flex-1 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-[15px] outline-none focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 dark:border-zinc-700 dark:bg-zinc-900"
              disabled={!datasetId}
              aria-label="Your question"
            />
            <Button type="submit" disabled={!input.trim() || !!pending || !datasetId}>
              {pending ? <Spinner /> : "Ask"}
            </Button>
          </div>
        </form>
      </section>
    </div>
  );
}
