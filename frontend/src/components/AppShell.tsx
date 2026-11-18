"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { Suspense, useEffect, useState, type ReactNode } from "react";

import { Spinner } from "@/components/ui";
import { api } from "@/lib/api";
import type { User } from "@/lib/types";

const NAV = [
  { href: "/chat", label: "Ask" },
  { href: "/datasets", label: "Datasets" },
];

/** Free hosting (Render) sleeps when idle; the first request can take ~30-60s. */
function useBackendWaking(): boolean {
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    let done = false;
    const timer = setTimeout(() => !done && setWaking(true), 3000);
    api("/health")
      .catch(() => undefined)
      .finally(() => {
        done = true;
        clearTimeout(timer);
        setWaking(false);
      });
    return () => clearTimeout(timer);
  }, []);
  return waking;
}

/** Reads the URL (to highlight the active link), so it renders inside <Suspense>. */
function NavLinks() {
  return <NavList active={usePathname()} />;
}

function NavList({ active }: { active: string | null }) {
  return (
    <nav className="flex gap-1">
      {NAV.map((item) => {
        const isActive = active?.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              isActive
                ? "bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-white"
                : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-white"
            }`}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const waking = useBackendWaking();

  useEffect(() => {
    api<User>("/auth/me").then(setUser).catch(() => undefined);
  }, []);

  async function logout() {
    await api("/auth/logout", { method: "POST" }).catch(() => undefined);
    router.replace("/login");
    router.refresh();
  }

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-20 border-b border-zinc-200 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/80">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-6 px-4">
          <Link href="/chat" className="flex items-center gap-2 font-semibold">
            <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-600 text-sm font-bold text-white">
              A
            </span>
            AskData
          </Link>
          <Suspense fallback={<NavList active={null} />}>
            <NavLinks />
          </Suspense>
          <div className="ml-auto flex items-center gap-3 text-sm">
            {user && <span className="hidden text-zinc-500 sm:inline">{user.email}</span>}
            <button
              onClick={logout}
              className="rounded-lg px-2.5 py-1.5 text-zinc-600 hover:bg-zinc-100 dark:text-zinc-400 dark:hover:bg-zinc-800"
            >
              Log out
            </button>
          </div>
        </div>
        {waking && (
          <div className="flex items-center justify-center gap-2 border-t border-amber-200 bg-amber-50 px-4 py-2 text-sm text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            <Spinner />
            Waking up the server. On the free plan this can take up to a minute after a quiet
            period.
          </div>
        )}
      </header>
      <div className="flex flex-1 flex-col">{children}</div>
    </div>
  );
}
