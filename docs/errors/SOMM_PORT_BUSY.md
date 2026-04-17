# SOMM_PORT_BUSY

**Problem.** `somm serve` (or `somm-serve`) tried to bind to a port that's
already in use.

**Why.** Another somm service instance, a stale uvicorn process, or an
unrelated app holds the port.

**Fix.**

1. Check if a somm service is already running:
   ```bash
   somm doctor   # scans for active somm processes + cooldowns
   ```

2. Pick a different port:
   ```bash
   somm serve --port 7879
   ```

3. Find what's holding the default port:
   ```bash
   lsof -i :7878           # macOS / Linux
   ss -tlnp | grep :7878   # Linux
   ```

   Kill the holder if it's a stale somm process:
   ```bash
   kill <pid>
   ```

4. If you'd like a permanent non-default port, set it in `somm.toml`:
   ```toml
   [tool.somm.service]
   port = 7879
   ```

   (Config support lands alongside `somm init` — see CHANGELOG.)

**Note.** `somm serve` binds to `127.0.0.1` by default. If you're
intentionally exposing to LAN via `--bind 0.0.0.0`, read the warning
in the server output carefully — traces are stored in plain SQLite on
disk and the admin UI has no auth.
