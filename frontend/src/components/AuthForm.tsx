"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Button, Card, ErrorNote, Input, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

/** Only allow redirects back into this app (no open redirects). */
function safeNext(next: string | null): string {
  return next && next.startsWith("/") && !next.startsWith("//") ? next : "/chat";
}

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const params = useSearchParams();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    try {
      if (mode === "signup") {
        await api("/auth/signup", {
          json: {
            email: data.get("email"),
            password: data.get("password"),
            workspace_name: data.get("workspace"),
          },
          allowUnauthorized: true,
        });
      } else {
        await api("/auth/login", {
          form: new URLSearchParams({
            username: String(data.get("email")),
            password: String(data.get("password")),
          }),
          allowUnauthorized: true,
        });
      }
      router.replace(mode === "signup" ? "/datasets" : safeNext(params.get("next")));
      router.refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong.");
      setBusy(false);
    }
  }

  const signup = mode === "signup";
  return (
    <main className="flex flex-1 items-center justify-center px-4 py-12">
      <div className="w-full max-w-sm">
        <div className="mb-6 text-center">
          <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-xl bg-indigo-600 text-lg font-bold text-white">
            A
          </div>
          <h1 className="text-xl font-semibold">
            {signup ? "Create your workspace" : "Log in to AskData"}
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            {signup
              ? "Upload a spreadsheet and ask questions in plain English."
              : "Ask questions about your business data."}
          </p>
        </div>
        <Card className="p-6">
          <form onSubmit={onSubmit} className="space-y-4">
            {signup && (
              <label className="block space-y-1.5">
                <span className="text-sm font-medium">Company name</span>
                <Input name="workspace" required maxLength={120} placeholder="Acme Ltd" />
              </label>
            )}
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Email</span>
              <Input name="email" type="email" required autoComplete="email" />
            </label>
            <label className="block space-y-1.5">
              <span className="text-sm font-medium">Password</span>
              <Input
                name="password"
                type="password"
                required
                minLength={signup ? 8 : undefined}
                autoComplete={signup ? "new-password" : "current-password"}
              />
              {signup && <span className="text-xs text-zinc-500">At least 8 characters.</span>}
            </label>
            {error && <ErrorNote>{error}</ErrorNote>}
            <Button type="submit" disabled={busy} className="w-full">
              {busy && <Spinner />}
              {signup ? "Create workspace" : "Log in"}
            </Button>
          </form>
        </Card>
        <p className="mt-4 text-center text-sm text-zinc-500">
          {signup ? "Already have an account? " : "New to AskData? "}
          <Link
            href={signup ? "/login" : "/signup"}
            className="font-medium text-indigo-600 hover:underline dark:text-indigo-400"
          >
            {signup ? "Log in" : "Create a workspace"}
          </Link>
        </p>
      </div>
    </main>
  );
}
