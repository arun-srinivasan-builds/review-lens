# Deploying ReviewLens to a VPS

Two sites, one image, one Caddy proxy:

| Hostname | Mode | Access | OpenAI |
|----------|------|--------|--------|
| `reviewlens.arunsrinivasan.io` | `demo` | public, no login | never called; container has no key |
| `reviewlens-admin.arunsrinivasan.io` | `live` | password (Caddy basic auth) | real analysis |

Only Caddy publishes ports (80/443). The app containers run read-only, as a non-root user, with all
capabilities dropped, and are reachable only on the internal Docker network.

## 0. One-time, on your own machine

1. Build the demo bundle once (spends ~207 OpenAI calls; needs `.env` with key and model):

   ```bash
   .venv\Scripts\python -m scripts.build_demo
   ```

   This writes `demo_data/enriched.csv` and `demo_data/report.json`. Open the app locally with
   `REVIEWLENS_MODE=demo` to check it looks right, then commit `demo_data/`. It contains only the synthetic
   sample reviews, no secrets.
2. Make sure the demo bundle is committed and the repo is pushed or archived (see step 3).

## 1. DNS

Create `A` (and `AAAA` if the VPS has IPv6) records for **both** hostnames pointing at the VPS IP. Caddy
cannot obtain certificates until DNS resolves.

## 2. VPS prerequisites

- Ubuntu/Debian with Docker Engine and the Compose plugin ([official install guide](https://docs.docker.com/engine/install/)).
- Firewall: allow 22, 80, 443 (and 443/udp for HTTP/3); nothing else.

  ```bash
  sudo ufw allow OpenSSH && sudo ufw allow 80,443/tcp && sudo ufw allow 443/udp && sudo ufw enable
  ```

## 3. Get the code onto the VPS

```bash
git clone https://github.com/arun-srinivasan-builds/review-lens.git
cd review-lens
```

## 4. Secrets (on the VPS only, never committed)

```bash
cp .env.example .env
nano .env            # set OPENAI_API_KEY and OPENAI_MODEL
chmod 600 .env

cp caddy.env.example caddy.env
docker run --rm caddy:2 caddy hash-password --plaintext 'a-long-random-password'
nano caddy.env       # set ADMIN_USER and paste the hash inside single quotes
chmod 600 caddy.env
```

Use a long random password (20+ characters): Caddy basic auth has no built-in lockout.

## 5. Launch

```bash
docker compose build
docker compose up -d
docker compose ps          # demo, live, caddy should be running/healthy
docker compose logs -f caddy   # watch certificates being issued
```

## 6. Verify

- `https://reviewlens.arunsrinivasan.io` shows the dashboard with no upload box and no "Analyse" button.
- `https://reviewlens-admin.arunsrinivasan.io` asks for the password, then shows the upload page.
- The demo container has no key: `docker compose exec demo env | grep -c OPENAI` prints `0`.
- Upload `sample_data/sample_reviews.csv` on the admin site and analyse it.

## Operations

| Task | Command |
|------|---------|
| Update | `git pull && docker compose build && docker compose up -d` |
| Logs | `docker compose logs -f live` |
| Rotate OpenAI key | edit `.env`, then `docker compose up -d live` |
| Change admin password | new hash into `caddy.env`, then `docker compose up -d caddy` |
| Stop everything | `docker compose down` |

## Cost and safety notes

- Set a monthly spend limit on the OpenAI project that owns the key.
- The live site is the only place that can spend quota. It is behind the password and a 12 MB request cap.
- Uploads are processed in memory only; nothing is stored on the VPS. Review text is sent to OpenAI.
- Logs contain counts and exception names only, never review text or keys.
