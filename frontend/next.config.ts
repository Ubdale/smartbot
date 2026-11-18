import type { NextConfig } from "next";

// The browser only ever talks to this app. Requests to /api/* are forwarded to the
// FastAPI backend (Render in production), so the session cookie is first-party and
// no CORS is needed. Set BACKEND_URL in Vercel's project settings.
const backendUrl = (process.env.BACKEND_URL ?? "http://localhost:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  experimental: {
    // Next's rewrite proxy gives up after 30s by default; a local LLM on a CPU can take
    // a minute or more to answer. (On Vercel, the platform's own proxy limits apply.)
    proxyTimeout: 300_000,
    // Next buffers every request body up to this size before rewrites forward it, and
    // anything larger arrives truncated (default 10 MB). Uploads are capped at 20 MB by
    // the backend, plus multipart overhead.
    proxyClientMaxBodySize: "25mb",
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backendUrl}/:path*` }];
  },
};

export default nextConfig;
