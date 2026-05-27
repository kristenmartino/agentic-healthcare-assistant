/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The Vercel build pulls NEXT_PUBLIC_API_BASE from the project's env so
  // the production frontend talks to the Fly.io backend. Locally, .env.local
  // points at http://localhost:8000.
  env: {
    NEXT_PUBLIC_APP_NAME: "Healthcare Assistant",
  },
};

export default nextConfig;
