# somm-mcp

MCP server for [somm](https://github.com/lavallee/somm) — lets
MCP-capable coding agents (Claude Code, Cursor, Windsurf) query your
real LLM telemetry and the cross-project sommelier.

Ships 10 stdio tools: somm_stats, somm_search_calls, somm_recommend,
somm_register_workload, somm_register_prompt, somm_compare,
somm_replay, somm_advise, somm_record_decision, somm_search_decisions.

```json
{ "command": "somm-mcp", "env": { "SOMM_PROJECT": "my_app" } }
```
