# somm-service

The service tier for [somm](https://github.com/lavallee/somm) — the
self-hosted LLM telemetry, routing, and intelligence loop.

Adds the localhost web admin, the HTTP API, and the background
scheduler with three workers: model-intel refresh (pricing + context
windows + capabilities), online evaluation (samples production calls
and grades them against a gold model), and the agent worker (turns
telemetry + eval results into concrete recommendations).

```bash
pip install somm somm-service
somm serve --project my_app   # dashboard at localhost:7878
```
