# somm-core

Shared foundation for [somm](https://github.com/lavallee/somm) — the
self-hosted LLM telemetry, routing, and intelligence loop.

Contains the SQLite schema + migrations, the repository layer, config
loading, parse helpers (multimodal content blocks, capability
inference, JSON extraction), pricing (including the bundled offline
pricing snapshot), and the version constants every somm package
imports.

You usually want to install [`somm`](https://pypi.org/project/somm/)
(which depends on this package) rather than somm-core directly.
