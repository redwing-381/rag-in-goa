/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The mic requires a secure context. localhost counts as secure, so dev works,
  // but any other host must be served over HTTPS or getUserMedia is unavailable.
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000",
  },
};

export default nextConfig;
