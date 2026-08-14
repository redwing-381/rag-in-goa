# Deploy

The live link is a Fly.io API plus a Vercel frontend. The ~650 MB index
is baked into the Docker image at `fly deploy` time from this laptop;
it is not in git.

## API

Needs the [Fly CLI](https://fly.io/docs/flyctl/install/) and an account.

```bash
# once
fly auth login
fly launch --no-deploy --copy-config   # uses fly.toml (4 GB VM, Singapore)
fly secrets set OPENROUTER_API_KEY=... SARVAM_API_KEY=...
fly deploy                             # uploads image + index, ~10 min
```

Health check: `https://<app>.fly.dev/health` should report
`chunks: 215608` and `ready: true`. First boot is slow (index mmap + warmup).

## Frontend

```bash
cd apps/web
npx vercel --prod
```

Set `NEXT_PUBLIC_API_URL` to `https://<app>.fly.dev` in the Vercel project
settings and redeploy so the browser calls Fly, not localhost.

CORS on the API is `*` by default (`RAGOA_CORS_ORIGINS`). Tighten it to
the Vercel origin for production if you want.

## What not to upload

`.env`, `data/raw`, `data/slim`. The image only needs `data/index` and
`data/onnx`.
