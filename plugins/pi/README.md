# memsearch for pi

Automatic semantic memory for pi — remembers what you worked on across sessions.

## Quick Start

```bash
# Install globally
#pi install @zilliz/memsearch-pi

# Or install locally (for development/testing)
ln -s "$(pwd)/plugins/pi/memsearch" ~/.pi/agent/extensions/memsearch
```

## What It Does

- **Background indexing**: Watches your `.memsearch/memory/` directory and indexes markdown files into a Milvus vector store for semantic search
- **Cold-start context**: On first prompt of each session, injects recent memory headings and bullet points so pi knows what you were working on
- **Semantic recall tools**: Registers `memsearch_search` and `memsearch_expand` tools so pi can deep-search memory on demand
- **Session summarization**: `/memsearch-summarize` command summarizes current conversation and saves to daily `.md`

## Architecture

```
Session start
  ├─ Derive per-project collection name (ms_<name>_<hash>)
  ├─ Default to ONNX embedding provider (zero-config, CPU-only)
  ├─ Server mode: start persistent memsearch watch
  └─ Lite mode: one-time background index

User sends prompt
  ├─ Cold-start (once per session): inject recent memory context
  └─ pi uses memsearch_search / memsearch_expand tools as needed

/memsearch-summarize
  ├─ Extract conversation transcript
  ├─ Summarize with LLM (or raw text fallback)
  ├─ Append to .memsearch/memory/YYYY-MM-DD.md
  └─ Re-index

Session shutdown
  └─ Kill watch process
```

## Configuration

memsearch uses its own TOML config files:

- `~/.memsearch/config.toml` — user global
- `.memsearch.toml` — project local

If no config exists, defaults to ONNX (bge-m3, CPU-only, no API key needed).

To use another provider:

```bash
memsearch config set embedding.provider openai
```

See [memsearch docs](https://github.com/zilliztech/memsearch) for all configuration options.

## Requirements

- Node.js (pi runtime)
- Python 3.10+ with `memsearch[onnx]` installed, or `uv` (auto-detected)
- For ONNX provider: ~2GB RAM for model loading
- For cloud providers: appropriate API key in environment
