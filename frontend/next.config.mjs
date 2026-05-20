/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  async rewrites() {
    const apiBase = process.env.FORGE_API_URL ?? "http://127.0.0.1:8000";
    return [
      {
        source: "/api/control/:path*",
        destination: `${apiBase}/api/control/:path*`
      },
      {
        source: "/healthz",
        destination: `${apiBase}/healthz`
      }
    ];
  }
};

export default nextConfig;
