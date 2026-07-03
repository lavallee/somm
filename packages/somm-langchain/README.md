# somm-langchain

LangChain `BaseChatModel` adapter for [`somm`](../somm).

Lets LangChain / LangGraph / `deepagents` apps use somm as their LLM substrate — telemetry, routing, cost tracking, provider fallback, sommelier model memory — all without changing the agent-framework call sites.

```python
import somm
from somm_langchain import SommChatModel

llm = somm.llm(project="my_agent")
chat = SommChatModel(somm_llm=llm, workload="agent_thinking")

from deepagents import create_deep_agent
agent = create_deep_agent(model=chat, tools=[...])
```

`SommChatModel` supports `bind_tools()` and routes tool-using calls through somm's neutral tool-calling shape (see [`../../docs/tool-calling.md`](../../docs/tool-calling.md)). The adapter is thin: message translation in, somm.llm().generate() through, ChatGeneration out.

Built for agent orchestrators that run on `deepagents` and similar LangChain-based stacks, which mandate a tool-calling `BaseChatModel`.
