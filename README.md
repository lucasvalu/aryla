# aryla v0.1.0 - Local, privacy-first assistant powered by LangGraph 

A terminal assistant that routes what you ask to the right specialist. It runs on LangGraph and a
local Ollama model, and it can read your code through the filesystem MCP server.

- **Orchestrator** decides who handles the request, using structured output from the model.
- **Coding agent** takes everything about code: explaining, writing, reviewing, debugging, and
  reading files from a workspace you point it at. Read-only for now.
- **General agent** takes everything else. No tools.
- Conversations are stored per `thread_id` in SQLite, with older messages folded into a summary.
- Any write tool would pause the graph and ask you first. None are enabled yet.

```
                    -> coding  -> filesystem MCP tools -> your workspace
you -> orchestrator -> general
                    -> clarify
                    -> answer
```

## Requirements

- [uv](https://docs.astral.sh/uv/). It downloads Python 3.12 on its own.
- [Ollama](https://ollama.com) running locally, with `ollama pull qwen3:1.7b`.
- Node.js with `npx` (Node 22 works) for the filesystem MCP server.

## Setup

```bash
uv sync
cp .env.example .env
```

Open `.env` and set `CODING_WORKSPACE` to the absolute path of the repository you want the coding
agent to read. Leave it empty and aryla still runs, with the general agent only.

## Run

```bash
uv run aryla                     # new conversation
uv run aryla --verbose           # also print what the graph is doing
uv run aryla --debug             # everything, including message content
```

Inside the chat: `/state`, `/new`, `/exit`, `/help`.

By default you only see your messages and the answers. With `--verbose` each step shows up on stderr:

```
12:29:12 | orchestrator -> coding (confidence 0.95, 6.1s): asks what a file does
12:29:12 | handoff: orchestrator -> coding agent (1 messages in context)
12:29:12 | coding: calling the model (round 1)
12:29:35 | coding: model wants 1 tool call(s) in 23.3s: read_text_file
12:29:35 | coding: running read_text_file({'path': '/home/me/repo/aryla/graph/builder.py'})
12:29:35 | coding: read_text_file returned success in 0.1s (3081 chars)
12:29:35 | coding: calling the model (round 2)
12:30:02 | coding: model answered in 27.1s
12:30:02 | handoff: coding agent -> consolidate (507 chars)
12:30:02 | turn: finished in 56.7s
```

On a CPU, `qwen3:1.7b` takes a few seconds per call, more on the first one while the model loads.
A coding turn that uses tools makes at least two calls.

`uv run langgraph dev` serves the graph from `langgraph.json` and opens LangGraph Studio, where you can
see the nodes, step through a run and inspect the state at each checkpoint. The UI needs a free
LangSmith login; the data stays on your machine. The dev server keeps its own state, separate from
the SQLite file the CLI uses.

## Configuration

| File | What it controls |
|---|---|
| `.env` | `CODING_WORKSPACE`, Ollama URL, and the `APP_*` knobs: confidence threshold, memory window, tool iteration limit, SQLite path |
| `config/models.yaml` | which model each role uses: `orchestrator`, `coding`, `general`, `summarizer` |
| `config/mcp.yaml` | MCP servers and the tool policy |

Models are plain config. To try another one, `ollama pull` it, change `models.yaml`, and run the
routing eval below. The `coding` role is the obvious place for a code model later. Keep `num_ctx`
at 8192 or more; Ollama's default is smaller and truncates silently. Other providers work through
`provider`/`model` as long as their LangChain package is installed; `aryla/llm/factory.py` is the
only file that knows about providers.

The tool policy is an allowlist. Tools under `read` run directly, tools under `write` pause for a
`y/N` confirmation in the terminal, and anything else never reaches the model. `write` is empty on
purpose; add `write_file` or `edit_file` there when you want the coding agent to change files.

## Tests

```bash
uv run pytest                            # unit + integration, fake model, in-process MCP server
uv run pytest -m eval                    # routing accuracy against the real model (slow)
uv run python scripts/routing_eval.py    # same thing, with a line per case
uv run ruff check . && uv run pyright
```

`tests/evals/routing_cases.yaml` has 52 messages with the agent each one should reach. Last run
with `qwen3:1.7b`: 92%. When routing gets something wrong in real use, add the message there.

## Layout

The package follows the usual LangGraph project shape: a root `graph.py`, one module per agent,
prompts and tools in their own packages. Start at `cli.py` and follow the calls down.

```
langgraph.json           graph registered for `langgraph dev`
config/                  models.yaml, mcp.yaml
src/aryla/
  cli.py                 main(): parses args, opens the graph, runs the chat loop
  graph.py               build_graph(): orchestrator -> agents | clarify -> consolidate -> summarize
  runtime.py             Runtime: settings, one model per role, tools and policy
  state.py               graph state and the records stored in it
  config.py              .env and YAML loading
  llm.py                 chat model per role (the only place that knows providers)
  memory.py              SQLite checkpointer and the rolling summary
  observability.py       the --verbose flow log and optional Langfuse tracing
  messages.py            text helpers for LangChain messages
  agents/
    __init__.py          the list of agents the orchestrator can pick
    base.py              Agent: name, description, examples, run()
    orchestrator.py      Orchestrator: decide -> route -> (agent | clarify)
    coding.py            CodingAgent: ReAct loop over the filesystem tools
    general.py           GeneralAgent: one model call
  prompts/               one .md per prompt, loaded with prompts.load("name")
  tools/
    policy.py            allowlist: read / write / blocked
    mcp.py               MCP server config and tool loading
scripts/                 routing_eval.py
tests/                   unit, integration, evals, fakes
```

A few rules keep it that way: dependencies point one way (`cli` -> `graph` -> `agents`/`memory` -> `tools`/`llm`/`prompts` -> `state`/`config`); nodes only talk through the state; everything an agent
needs comes in through `Runtime`; each agent owns its nodes, edges and `build()`; prompts are files, not code.
