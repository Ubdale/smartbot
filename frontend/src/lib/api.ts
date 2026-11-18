// Browser -> /api/* (same origin) -> FastAPI. The HttpOnly session cookie rides along
// automatically; writes add the CSRF header the backend requires for cookie auth.

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Options = Omit<RequestInit, "body"> & {
  json?: unknown;
  form?: FormData | URLSearchParams;
  /** Don't bounce to /login on 401 (used by the login form itself). */
  allowUnauthorized?: boolean;
};

function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    // Pydantic validation errors: [{loc: [...], msg: "..."}]
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => {
        const field = d.loc?.filter((p) => p !== "body").join(".");
        return field ? `${field}: ${d.msg}` : d.msg;
      })
      .join("; ");
  }
  if (status === 502 || status === 504) return "The server is unreachable right now. Try again in a moment.";
  return `Request failed (${status}).`;
}

export async function api<T>(path: string, opts: Options = {}): Promise<T> {
  const { json, form, allowUnauthorized, ...init } = opts;
  const method = (init.method ?? (json !== undefined || form ? "POST" : "GET")).toUpperCase();
  const headers = new Headers(init.headers);
  if (!["GET", "HEAD"].includes(method)) headers.set("X-AskData-CSRF", "1");

  let body: BodyInit | undefined;
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  } else if (form) {
    body = form; // the browser sets the multipart/urlencoded content type
  }

  let res: Response;
  try {
    res = await fetch(`/api${path}`, { ...init, method, headers, body, credentials: "same-origin" });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }

  if (res.status === 401 && !allowUnauthorized && typeof window !== "undefined") {
    // Outside React (no router here): a full navigation also clears any stale client state.
    const login = new URL("/login", window.location.origin);
    login.searchParams.set("next", window.location.pathname + window.location.search);
    window.location.assign(login.toString());
    throw new ApiError(401, "Your session has expired. Please log in again.");
  }
  if (res.status === 204) return undefined as T;

  const data: unknown = await res.json().catch(() => null);
  if (!res.ok) throw new ApiError(res.status, errorMessage(res.status, data));
  return data as T;
}
