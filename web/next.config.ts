import type { NextConfig } from "next";

// The browser calls same-origin /api/*; Next.js proxies it to the real backend.
// This keeps the backend URL server-side (no client inlining / CORS needed).
const backend = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default nextConfig;
