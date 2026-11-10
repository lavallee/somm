# SOMM_PROVIDER_NOT_CONFIGURED

**Problem.** A call pinned `provider=` to a provider this machine has not
configured, so there was nothing to send it to.

**Why.** Common triggers:

- The provider's API key is not set in the environment the process runs
  in (a systemd unit or container often lacks the shell's variables).
- A seat executor (`claude-cli`, `codex-cli`) was pinned but its binary is
  not on the process's `PATH`. The message names the directories somm
  searched.
- The provider name is misspelled.

**Behavior.** Raised as `somm.SommProviderNotConfigured` before any network
call, with `provider` set to the name that was asked for. It is also a
`ValueError`, which is what this case raised before it had a type.

**Fix.**

1. Set the key the provider reads (for example `MINIMAX_API_KEY`) where the
   process runs, or point somm at the binary with `SOMM_CLAUDE_BIN` /
   `SOMM_CODEX_BIN`.
2. Check what this machine has: `llm.health()` lists every provider with
   `configured` true or false.
3. If any provider will do, drop the pin and let the chain route.

**Related.**
- [`SOMM_PROVIDER_AUTH`](./SOMM_PROVIDER_AUTH.md) — the key is set but the
  provider rejects it.
