import { NextResponse, type NextRequest } from "next/server";

// Only checks that a session cookie exists, to route people to /login quickly.
// The backend verifies the token on every API call; this is not a security boundary.
const SESSION_COOKIE = "askdata_session";
const AUTH_PAGES = ["/login", "/signup"];

export function proxy(request: NextRequest) {
  const { pathname, search } = request.nextUrl;
  const hasSession = request.cookies.has(SESSION_COOKIE);
  const isAuthPage = AUTH_PAGES.some((p) => pathname === p || pathname.startsWith(`${p}/`));

  if (!hasSession && !isAuthPage) {
    const url = new URL("/login", request.url);
    if (pathname !== "/") url.searchParams.set("next", pathname + search);
    return NextResponse.redirect(url);
  }
  if (hasSession && isAuthPage) {
    return NextResponse.redirect(new URL("/chat", request.url));
  }
  return NextResponse.next();
}

export const config = {
  // Never run on /api: proxy buffers request bodies (10 MB default), which would
  // truncate dataset uploads. Also skip Next internals and static files.
  matcher: ["/((?!api/|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|ico)$).*)"],
};
