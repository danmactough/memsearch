/**
 * memsearch Extension for pi
 *
 * Automatic semantic memory across pi sessions.
 *
 * What it does:
 * - Indexes markdown memory files (watch or one-time depending on Milvus mode)
 * - Injects recent memory context at session start (cold-start)
 * - Registers memsearch_search and memsearch_expand tools for LLM-driven recall
 * - /memsearch-summarize command: summarize conversation, save to daily .md, re-index
 *
 * Installation:
 *   pi install git:github.com/danmactough/pi-memsearch
 */

import type { ExtensionAPI, SessionEntry } from "@mariozechner/pi-coding-agent";
import { SessionManager } from "@mariozechner/pi-coding-agent";
import { Type } from "typebox";
import { complete } from "@mariozechner/pi-ai";
import * as cp from "node:child_process";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ── Module-level state ──────────────────────────────────────────────

let runner: { cmd: string; prefixArgs: string[] } | null = null;
let collectionName: string;
let memoryDir: string;
let memsearchDir: string;
let coldStartInjected = false;
let watchProcess: cp.ChildProcess | null = null;

// ── Helpers ─────────────────────────────────────────────────────────

function deriveCollectionName(cwd: string): string {
  const basename = path
    .basename(cwd)
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "_")
    .replace(/__+/g, "_")
    .replace(/^_|_$/g, "")
    .slice(0, 40);
  const hash = crypto.createHash("sha256").update(cwd).digest("hex").slice(0, 8);
  return `ms_${basename}_${hash}`;
}

async function detectRunner(): Promise<{ cmd: string; prefixArgs: string[] } | null> {
  // Check memsearch on PATH
  try {
    const r = cp.execSync("which memsearch 2>/dev/null || true", { encoding: "utf-8" }).trim();
    if (r) return { cmd: "memsearch", prefixArgs: [] };
  } catch {
    // ignore
  }

  // Check uvx on PATH
  try {
    const r = cp.execSync("which uvx 2>/dev/null || true", { encoding: "utf-8" }).trim();
    if (r) return { cmd: "uvx", prefixArgs: ["--from", "memsearch[onnx]", "memsearch"] };
  } catch {
    // ignore
  }

  return null;
}

async function runMemsearch(
  pi: ExtensionAPI,
  args: string[],
  opts?: { timeout?: number; silent?: boolean },
): Promise<{ stdout: string; stderr: string; code: number | null }> {
  const allArgs = [...runner!.prefixArgs, ...args];
  if (collectionName) {
    allArgs.push("--collection", collectionName);
  }
  return pi.exec(runner!.cmd, allArgs, { timeout: opts?.timeout ?? 30_000 });
}

function spawnDetached(args: string[]): void {
  const allArgs = [...runner!.prefixArgs, ...args];
  if (collectionName) {
    allArgs.push("--collection", collectionName);
  }
  cp.spawn(runner!.cmd, allArgs, {
    stdio: "ignore",
    detached: true,
    cwd: process.cwd(),
  }).unref();
}

function ensureMemoryDir(): void {
  fs.mkdirSync(memoryDir, { recursive: true });
}

/**
 * Extract a readable conversation transcript from session entries.
 * Returns a string with user questions, assistant responses, and tool calls.
 */
function buildConversationText(entries: SessionEntry[]): string {
  const sections: string[] = [];

  for (const entry of entries) {
    if (entry.type !== "message") continue;
    const msg = entry.message;
    if (!msg?.role) continue;

    const role = msg.role;
    if (role !== "user" && role !== "assistant") continue;

    const content = msg.content;
    const textParts = extractTextParts(content);
    if (textParts.length === 0) continue;

    const roleLabel = role === "user" ? "User" : "pi";
    sections.push(`${roleLabel}: ${textParts.join(" ").trim()}`);

    // Capture tool calls from assistant messages
    if (role === "assistant") {
      const toolLines = extractToolCalls(content);
      sections.push(...toolLines);
    }
  }

  return sections.join("\n\n");
}

function extractTextParts(content: unknown): string[] {
  if (typeof content === "string") return [content];
  if (!Array.isArray(content)) return [];

  const parts: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as Record<string, unknown>;
    if (b.type === "text" && typeof b.text === "string") {
      parts.push(b.text);
    }
  }
  return parts;
}

function extractToolCalls(content: unknown): string[] {
  if (!Array.isArray(content)) return [];

  const calls: string[] = [];
  for (const block of content) {
    if (!block || typeof block !== "object") continue;
    const b = block as Record<string, unknown>;
    if (b.type !== "toolCall" || typeof b.name !== "string") continue;
    const args = b.arguments ?? {};
    calls.push(`Tool called: ${b.name}(${JSON.stringify(args).slice(0, 200)})`);
  }
  return calls;
}

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function getTime(): string {
  return new Date().toTimeString().slice(0, 5);
}

// ── Main extension ──────────────────────────────────────────────────

export default async function (pi: ExtensionAPI) {
  const cwd = process.cwd();

  // Detect memsearch
  runner = await detectRunner();
  if (!runner) {
    pi.on("session_start", async (_event, ctx) => {
      if (ctx.hasUI) {
        ctx.ui.notify(
          "[memsearch] not found. Install with: pip install 'memsearch[onnx]' or uv tool install 'memsearch[onnx]'",
          "warning",
        );
      }
    });
    return;
  }

  collectionName = deriveCollectionName(cwd);
  memsearchDir = path.join(cwd, ".memsearch");
  memoryDir = path.join(memsearchDir, "memory");

  // ── session_start: initialize ──────────────────────────────────

  pi.on("session_start", async (_event, ctx) => {
    coldStartInjected = false;
    ensureMemoryDir();

    // Default to ONNX provider if no config exists (zero-config setup)
    const userConfig = path.join(os.homedir(), ".memsearch", "config.toml");
    const projectConfig = path.join(cwd, ".memsearch.toml");
    if (!fs.existsSync(userConfig) && !fs.existsSync(projectConfig)) {
      try {
        cp.execSync(
          `${runner!.cmd} ${[...runner!.prefixArgs, "config", "set", "embedding.provider", "onnx"].join(" ")}`,
          { stdio: "ignore" },
        );
      } catch {
        // config set may fail if memsearch version doesn't support it — ignore
      }
    }

    // Get provider/model for status display
    let provider = "onnx";
    let milvusUri = "";
    try {
      const p = await runMemsearch(pi, ["config", "get", "embedding.provider"], { silent: true });
      provider = p.stdout.trim() || "onnx";
    } catch {
      // ignore
    }
    try {
      const m = await runMemsearch(pi, ["config", "get", "milvus.uri"], { silent: true });
      milvusUri = m.stdout.trim();
    } catch {
      // ignore
    }

    const isServer = milvusUri.startsWith("http") || milvusUri.startsWith("tcp");

    // Stop any prior watch process
    if (watchProcess) {
      try {
        process.kill(-watchProcess.pid!, "SIGTERM");
      } catch {
        // already dead
      }
      watchProcess = null;
    }

    if (isServer) {
      // Server mode: start persistent watch
      watchProcess = cp.spawn(runner!.cmd, [
        ...runner!.prefixArgs,
        "watch",
        memoryDir,
        "--collection",
        collectionName,
      ], {
        stdio: "ignore",
        detached: true,
        cwd,
      });
      watchProcess.unref();
    } else {
      // Lite mode: one-time index (background, ONNX model load takes ~10s)
      spawnDetached(["index", memoryDir]);
    }

    // Show status
    if (ctx.hasUI) {
      ctx.ui.setStatus(
        "memsearch",
        `memsearch | ${provider} | ${isServer ? "server" : "lite"} | ${collectionName}`,
      );
      ctx.ui.notify(`[memsearch] ready — ${provider}/${isServer ? "server" : "lite"}`, "info");
    }
  });

  // ── Cold-start context injection ───────────────────────────────

  pi.on("before_agent_start", async (event) => {
    if (coldStartInjected) return;
    coldStartInjected = true;

    if (!fs.existsSync(memoryDir)) return;

    // Find the 2 most recent daily .md files
    const files = fs
      .readdirSync(memoryDir)
      .filter((f) => f.endsWith(".md"))
      .sort()
      .reverse()
      .slice(0, 2);

    if (files.length === 0) return;

    let context = "# Recent Memory\n\n";
    for (const file of files) {
      const content = fs.readFileSync(path.join(memoryDir, file), "utf-8");
      // Extract headings and bullet points (high signal density)
      const lines = content
        .split("\n")
        .filter((l) => /^(#{2,4} |- )/.test(l))
        .slice(0, 40);
      if (lines.length > 0) {
        context += `## ${file}\n${lines.join("\n")}\n\n`;
      }
    }

    // Write session heading to today's memory file
    const today = getToday();
    const now = getTime();
    const memoryFile = path.join(memoryDir, `${today}.md`);
    ensureMemoryDir();
    if (!fs.existsSync(memoryFile) || !fs.readFileSync(memoryFile, "utf-8").includes(`## Session ${now}`)) {
      fs.appendFileSync(memoryFile, `\n## Session ${now}\n\n`);
    }

    return {
      systemPrompt:
        event.systemPrompt +
        `\n\n${context}\n\n` +
        `Use memsearch_search to find specific past context from semantic memory. ` +
        `Use memsearch_expand to get full markdown sections. ` +
        `Search is preferred over reading raw .memsearch/memory/*.md files.`,
    };
  });

  // ── Tools ──────────────────────────────────────────────────────

  pi.registerTool({
    name: "memsearch_search",
    label: "Search Memory",
    description:
      "Search semantic memory for relevant past context. Use when the user's question could benefit from " +
      "historical context, past decisions, debugging notes, previous conversations, or project knowledge. " +
      "Use before reading raw .memsearch/memory/ files.",
    promptSnippet: "Search semantic memory for relevant past context and decisions",
    promptGuidelines: [
      "Use memsearch_search to retrieve relevant memories before reading raw .memsearch/memory/ files.",
      "Use memsearch_expand after memsearch_search to get full markdown sections for promising chunks.",
    ],
    parameters: Type.Object({
      query: Type.String({ description: "Search query to find relevant memories" }),
      top_k: Type.Optional(
        Type.Number({ description: "Number of results (default: 5)", minimum: 1, maximum: 20 }),
      ),
    }),
    async execute(_toolCallId, params) {
      const k = params.top_k ?? 5;
      try {
        const result = await runMemsearch(pi, [
          "search",
          params.query,
          "--top-k",
          String(k),
          "--json-output",
        ]);
        return {
          content: [
            {
              type: "text",
              text: result.stdout.trim() || "No results found.",
            },
          ],
          details: {},
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `memsearch search failed: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: {},
          isError: true,
        };
      }
    },
  });

  pi.registerTool({
    name: "memsearch_expand",
    label: "Expand Memory Chunk",
    description:
      "Expand a memory chunk hash to get the full markdown section with surrounding context. " +
      "Use after memsearch_search to get complete content from promising results.",
    promptSnippet: "Expand a chunk hash into full markdown context",
    parameters: Type.Object({
      chunk_hash: Type.String({ description: "The chunk hash from memsearch_search results to expand" }),
    }),
    async execute(_toolCallId, params) {
      try {
        const result = await runMemsearch(pi, ["expand", params.chunk_hash]);
        return {
          content: [
            {
              type: "text",
              text: result.stdout.trim() || "No expansion available.",
            },
          ],
          details: {},
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `memsearch expand failed: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: {},
          isError: true,
        };
      }
    },
  });

  // ── /memsearch-summarize command ───────────────────────────────

  pi.registerCommand("memsearch-summarize", {
    description: "Summarize current conversation and save to memory",
    handler: async (args, ctx) => {
      // Load summarize prompt (shared across all summarization calls)
      const promptPath = path.join(__dirname, "prompts", "summarize.txt");
      const systemPrompt = fs.existsSync(promptPath)
        ? fs.readFileSync(promptPath, "utf-8")
        : "You are a third-person note-taker. Summarize the transcript as 2-6 bullet points. " +
          "Write in third person. Output ONLY bullet points.";

      // Collect sessions to summarize
      const sessions: Array<{ label: string; date: string; conversationText: string }> = [];

      if (args === "all") {
        const listCwd = ctx.sessionManager.getCwd();
        const sessionDir = ctx.sessionManager.getSessionDir();
        ctx.ui.notify("Listing sessions...", "info");

        let allSessions;
        try {
          allSessions = await SessionManager.list(listCwd, sessionDir);
        } catch (err) {
          ctx.ui.notify(`Failed to list sessions: ${err instanceof Error ? err.message : String(err)}`, "error");
          return;
        }

        if (allSessions.length === 0) {
          ctx.ui.notify("No sessions found.", "warning");
          return;
        }

        for (const si of allSessions) {
          try {
            const sm = SessionManager.open(si.path, sessionDir, listCwd);
            const text = buildConversationText(sm.getEntries());
            if (text.trim()) {
              sessions.push({
                label: si.name || si.id.slice(0, 8),
                date: si.created.toISOString().slice(0, 10),
                conversationText: text,
              });
            }
          } catch (err) {
            ctx.ui.notify(
              `Failed to open session ${si.id.slice(0, 8)}: ${err instanceof Error ? err.message : String(err)}`,
              "error",
            );
          }
        }
      } else {
        const text = buildConversationText(ctx.sessionManager.getEntries());
        if (text.trim()) {
          sessions.push({ label: getTime(), date: getToday(), conversationText: text });
        }
      }

      if (sessions.length === 0) {
        ctx.ui.notify("No conversation to summarize.", "warning");
        return;
      }

      // Summarize each session, writing to the date file matching the session
      ensureMemoryDir();
      const touchedFiles = new Set<string>();

      let summarized = 0;
      for (let i = 0; i < sessions.length; i++) {
        const { label, date, conversationText } = sessions[i]!;
        if (sessions.length > 1) {
          ctx.ui.notify(`Summarizing ${i + 1}/${sessions.length}: ${label}...`, "info");
        } else {
          ctx.ui.notify("Summarizing conversation...", "info");
        }

        const summary = await summarizeText(
          conversationText,
          systemPrompt,
          ctx.model,
          ctx.modelRegistry,
          ctx.ui,
        );

        if (!summary.trim()) continue;

        const memoryFile = path.join(memoryDir, `${date}.md`);
        fs.appendFileSync(memoryFile, `### ${label}\n${summary}\n\n`);
        touchedFiles.add(date);
        summarized++;
      }

      if (summarized === 0) {
        ctx.ui.notify("Summarization produced empty output.", "warning");
        return;
      }

      // Re-index all saved summaries at once
      const msg = sessions.length > 1
        ? `Summarized ${summarized}/${sessions.length} sessions.`
        : "Summary complete.";
      const fileList = [...touchedFiles].sort().join(", ");
      try {
        await runMemsearch(pi, ["index", memoryDir], { timeout: 60_000 });
        ctx.ui.notify(`${msg} Saved to ${fileList} and re-indexed.`, "info");
      } catch {
        ctx.ui.notify(`${msg} Saved to ${fileList} (re-index failed — will be picked up by watch).`, "warning");
      }
    },
  });

  // ── session_shutdown: cleanup ──────────────────────────────────

  pi.on("session_shutdown", async () => {
    if (watchProcess) {
      try {
        process.kill(-watchProcess.pid!, "SIGTERM");
      } catch {
        // already dead
      }
      watchProcess = null;
    }
  });
};

// ── Fallback: format raw conversation as simple bullet points ──────

function formatRawBullets(text: string): string {
  const lines = text
    .split("\n")
    .filter((l) => l.trim())
    .slice(0, 20);
  return lines.map((l) => `- ${l.trim()}`).join("\n");
}

async function summarizeText(
  conversationText: string,
  systemPrompt: string,
  model: any,
  modelRegistry: any,
  ui: any,
): Promise<string> {
  if (!model) return formatRawBullets(conversationText);

  const auth = await modelRegistry.getApiKeyAndHeaders(model);
  if (!auth.ok || !auth.apiKey) return formatRawBullets(conversationText);

  try {
    const response = await complete(
      model,
      {
        messages: [
          {
            role: "user",
            content: [
              { type: "text", text: `${systemPrompt}\n\n<transcript>\n${conversationText}\n</transcript>` },
            ],
            timestamp: Date.now(),
          },
        ],
      },
      {
        apiKey: auth.apiKey,
        maxTokens: 2048,
        ...(auth.headers ? { headers: auth.headers } : {}),
      },
    );

    return response.content
      .filter((c): c is { type: "text"; text: string } => c.type === "text")
      .map((c) => c.text)
      .join("\n")
      .trim();
  } catch (err) {
    ui.notify(`LLM summarization failed: ${err instanceof Error ? err.message : String(err)}`, "error");
    return formatRawBullets(conversationText);
  }
}
