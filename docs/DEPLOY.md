# Deploying ReviewLens to a VPS (behind an existing Traefik)

Two sites, one image, routed by the Traefik reverse proxy already running on the VPS (the n8n stack's
`n8n-traefik-1`: entrypoint `websecure`, certificate resolver `mytlschallenge`, network `n8n_default`):

| Hostname | Mode | Access | OpenAI |
|----------|------|--------|--------|
| `reviewlens.arunsrinivasan.io` | `demo` | public, no login | never called; container has no key |
| `reviewlens-admin.arunsrinivasan.io` | `live` | password (Traefik basic auth) | real analysis |

The app containers publish no ports. They run read-only, as a non-root user, with all capabilities
dropped, and are reached only through Traefik. Traefik and n8n are not modified.

## 1. DNS

`A` records for both hostnames pointing at the VPS IP (done once).

## 2. Get the code

```bash
git clone https://github.com/arun-srinivasan-builds/review-lens.git
cd review-lens
```

## 3. Secrets (on the VPS only, never committed)

```bash
cp .env.example .env
nano .env     # set OPENAI_API_KEY and OPENAI_MODEL; leave OPENAI_TEMPERATURE empty
chmod 600 .env
```

Generate the admin password hash (it prompts for the password, so it stays out of shell history):

```bash
docker run --rm -it caddy:2 caddy hash-password
```

Add these two lines at the end of `.env`, pasting the hash between single quotes:

```env
ADMIN_USER=admin
ADMIN_PASSWORD_HASH='$2a$14$...the-hash-you-copied...'
```

The hash must be in single quotes because it contains `$` characters. Use a long random password.

## 4. Launch

```bash
docker compose build
docker compose up -d
docker compose ps
```

`demo` and `live` should be `Up (healthy)`. Traefik discovers them from their labels and requests the
certificates on the first visit to each hostname (a few seconds).

## 5. Verify

```bash
curl -sI https://reviewlens.arunsrinivasan.io | head -1          # 200
curl -sI https://reviewlens-admin.arunsrinivasan.io | head -1    # 401 (password required)
docker compose exec demo env | grep -c OPENAI                    # 0
```

Then open the admin site, sign in, upload `sample_data/sample_reviews.csv` and click **Analyse reviews**.

## Operations

| Task | Command |
|------|---------|
| Update | `git pull && docker compose build && docker compose up -d` |
| Logs | `docker compose logs -f live` |
| Rotate OpenAI key | edit `.env`, then `docker compose up -d live` |
| Change admin password | new hash into `.env`, then `docker compose up -d` |
| Stop ReviewLens only | `docker compose down` (does not affect Traefik or n8n) |

## Rebuilding the demo bundle (on your own machine)

```bash
python -m scripts.build_demo     # about 207 OpenAI calls
```

Commit the regenerated `demo_data/` and redeploy. It contains only the synthetic sample reviews.

## Safety notes

- Only the live site can call OpenAI. It is behind the password and a 12 MB request cap.
- Uploads are processed in memory only; nothing is stored on the VPS. Review text is sent to OpenAI.
- Logs contain counts and exception names only, never review text or keys.
