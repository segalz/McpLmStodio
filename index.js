import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import axios from "axios";
import fs from "fs";
import path from "path";
import { execSync } from "child_process";

// ── Filesystem tools definition (OpenAI format) ──
const FILESYSTEM_TOOLS = [
  {
    type: "function",
    function: {
      name: "list_directory",
      description: "List files and directories at a given path",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute directory path" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "read_file",
      description: "Read the text content of a file",
      parameters: {
        type: "object",
        properties: {
          path: { type: "string", description: "Absolute file path" },
        },
        required: ["path"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_files",
      description: "Search for files matching a glob pattern (e.g. **/*.md)",
      parameters: {
        type: "object",
        properties: {
          directory: { type: "string", description: "Base directory to search in" },
          pattern: { type: "string", description: "Glob-like pattern to match (e.g. *.md, **/*.js)" },
        },
        required: ["directory", "pattern"],
      },
    },
  },
  {
    type: "function",
    function: {
      name: "git_diff",
      description: "Run git diff to compare current uncommitted changes vs last commit for a specific file or all files. Returns the unified diff output.",
      parameters: {
        type: "object",
        properties: {
          file: { type: "string", description: "File path relative to project root (e.g. components/Feed.js). Leave empty for all changed files." },
        },
      },
    },
  },
];

// ── Execute filesystem tool calls locally ──
function executeToolCall(name, args) {
  try {
    if (name === "list_directory") {
      const entries = fs.readdirSync(args.path, { withFileTypes: true });
      return entries
        .map((e) => `${e.isDirectory() ? "[DIR]" : "[FILE]"} ${e.name}`)
        .join("\n");
    }

    if (name === "read_file") {
      const content = fs.readFileSync(args.path, "utf-8");
      // Limit to 10K chars to avoid blowing up context
      if (content.length > 10000) {
        return content.slice(0, 10000) + "\n\n... (truncated)";
      }
      return content;
    }

    if (name === "search_files") {
      const results = [];
      const regex = globToRegex(args.pattern);
      walkDir(args.directory, args.directory, regex, results, 0, 5);
      return results.length > 0
        ? results.join("\n")
        : "No files matched the pattern.";
    }

    if (name === "git_diff") {
      const cwd = process.env.GIT_PROJECT_PATH || "/Users/zvisegal/devlope/TaskYamGitMobile";
      const fileArg = args.file ? `-- ${args.file}` : "";
      const result = execSync(`git diff HEAD ${fileArg}`, { cwd, encoding: "utf-8", maxBuffer: 1024 * 1024 });
      if (!result.trim()) return "No uncommitted changes found.";
      if (result.length > 10000) return result.slice(0, 10000) + "\n\n... (truncated)";
      return result;
    }

    return `Unknown tool: ${name}`;
  } catch (err) {
    return `Error executing ${name}: ${err.message}`;
  }
}

// ── Helper: simple glob to regex ──
function globToRegex(pattern) {
  const escaped = pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, "<<DOUBLESTAR>>")
    .replace(/\*/g, "[^/]*")
    .replace(/<<DOUBLESTAR>>/g, ".*")
    .replace(/\?/g, ".");
  return new RegExp(`^${escaped}$`);
}

// ── Helper: recursive directory walk ──
function walkDir(baseDir, currentDir, regex, results, depth, maxDepth) {
  if (depth > maxDepth || results.length > 50) return;
  try {
    const entries = fs.readdirSync(currentDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.name.startsWith(".") || entry.name === "node_modules") continue;
      const fullPath = path.join(currentDir, entry.name);
      const relativePath = path.relative(baseDir, fullPath);
      if (entry.isDirectory()) {
        walkDir(baseDir, fullPath, regex, results, depth + 1, maxDepth);
      } else if (regex.test(relativePath) || regex.test(entry.name)) {
        results.push(fullPath);
      }
    }
  } catch (_) { /* skip unreadable dirs */ }
}

// ── Ollama config ──
const OLLAMA_BASE = "http://localhost:11434";
const OLLAMA_URL = `${OLLAMA_BASE}/v1/chat/completions`;
const OLLAMA_MODELS_URL = `${OLLAMA_BASE}/v1/models`;
const PREFERRED_MODEL = process.env.OLLAMA_MODEL || "qwen3-coder-next:latest";
const PREFERRED_DEEPSEEK = process.env.OLLAMA_DEEPSEEK_MODEL || "deepseek-r1:latest";
const MAX_STEPS = 10;
const LLM_TIMEOUT = 180_000; // 3 minutes per LLM call

// ── Auto-detect available model ──
// Queries Ollama /v1/models to find available models.
// Returns preferred if available, otherwise the first model found.
async function getLoadedModel(preferred) {
  try {
    const res = await axios.get(OLLAMA_MODELS_URL, { timeout: 5000 });
    const models = Array.isArray(res.data?.data) ? res.data.data : [];
    const available = models.map((m) => m.id);
    if (available.length === 0) throw new Error("No models available in Ollama");
    const match = available.find((id) => id === preferred);
    return match ?? available[0];
  } catch (err) {
    return preferred;
  }
}

// 1. Initialize the MCP Server
const server = new Server(
  { name: "LMStudio Docs, Git Review & Code Feedback Server", version: "1.3.0" },
  { capabilities: { tools: {} } }
);

// 2. Define the tool exposed by this server
server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "ask_lm_studio_docs",
      description:
        "Ask a specific question or summarize requirements from local markdown documentation via Ollama.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "The exact question or summarization request regarding the architecture or requirements found in the .md files.",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "git_review",
      description:
        "Review uncommitted code changes (git diff) using Ollama. Analyzes what changed, checks for bugs, regressions, and issues.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "What to review or check in the code changes. E.g. 'Review all changes for bugs and regressions' or 'Check components/Feed.js changes'.",
          },
          file: {
            type: "string",
            description:
              "Optional: specific file to review (e.g. 'components/Feed.js'). If omitted, reviews all changed files.",
          },
        },
        required: ["query"],
      },
    },
    {
      name: "code_feedback",
      description:
        "Get implementation feedback on uncommitted code changes using DeepSeek reasoning model. Provides comments on code quality, patterns, modularity, and maintainability — without suggesting rewrites.",
      inputSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description:
              "What to review or get feedback on. E.g. 'Review implementation quality' or 'Check if the new component follows project patterns'.",
          },
          file: {
            type: "string",
            description:
              "Optional: specific file to review (e.g. 'components/Feed.js'). If omitted, reviews all changed files.",
          },
        },
        required: ["query"],
      },
    },
  ],
}));

// 3. Tool execution — agentic loop with manual tool calling
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name === "ask_lm_studio_docs") {
    const query = request.params.arguments?.query;

    if (!query || typeof query !== "string") {
      return {
        content: [{ type: "text", text: "Invalid arguments: 'query' is required and must be a string." }],
        isError: true,
      };
    }

    try {
      let messages = [
        {
          role: "system",
          content:
            "You are an expert technical assistant with access to local filesystem tools. " +
            "Use the provided tools (list_directory, read_file, search_files) to find and read files, " +
            "then answer the user's question based on their content. " +
            "Always use tools first before answering — do NOT say you cannot access files. " +
            "The documentation files are located in this path: '/Users/zvisegal/Library/CloudStorage/OneDrive-IsraelPorts/MD'. " +
            "Start by listing or searching that directory to find relevant .md files.",
        },
        { role: "user", content: query },
      ];

      let finalContent = null;
      const activeModel = await getLoadedModel(PREFERRED_MODEL);

      for (let step = 0; step < MAX_STEPS; step++) {
        const response = await axios.post(OLLAMA_URL, {
          model: activeModel,
          messages,
          temperature: 0.1,
          tools: FILESYSTEM_TOOLS,
        }, { timeout: LLM_TIMEOUT });

        const choice = response.data?.choices?.[0];
        const message = choice?.message;

        if (!message) {
          return {
            content: [{ type: "text", text: "Ollama returned no readable message." }],
            isError: true,
          };
        }

        // Add assistant message to history
        messages.push(message);

        // If done — no tool calls, just text
        if (
          choice.finish_reason === "stop" ||
          (message.content && (!message.tool_calls || message.tool_calls.length === 0))
        ) {
          finalContent = message.content || "";
          break;
        }

        // Execute each tool call and add results
        if (message.tool_calls && message.tool_calls.length > 0) {
          for (const toolCall of message.tool_calls) {
            const fnName = toolCall.function.name;
            let fnArgs;
            try {
              fnArgs = JSON.parse(toolCall.function.arguments);
            } catch {
              fnArgs = {};
            }

            console.error(`[Step ${step + 1}] Tool call: ${fnName}(${JSON.stringify(fnArgs)})`);
            const result = executeToolCall(fnName, fnArgs);

            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: result,
            });
          }
        }
      }

      if (finalContent === null) {
        return {
          content: [{ type: "text", text: "Ollama reached max steps without a final answer." }],
          isError: true,
        };
      }

      return { content: [{ type: "text", text: finalContent }] };
    } catch (error) {
      let errorMessage = "Unknown error communicating with Ollama.";
      if (axios.isAxiosError(error)) {
        errorMessage = `Ollama Error: ${error.response?.data?.error?.message || error.message}. Is Ollama running on port 11434?`;
      } else if (error instanceof Error) {
        errorMessage = `Internal Error: ${error.message}`;
      }
      return { content: [{ type: "text", text: errorMessage }], isError: true };
    }
  }

  if (request.params.name === "git_review") {
    const query = request.params.arguments?.query;
    const file = request.params.arguments?.file || "";

    if (!query || typeof query !== "string") {
      return {
        content: [{ type: "text", text: "Invalid arguments: 'query' is required and must be a string." }],
        isError: true,
      };
    }

    try {
      // Get the diff first
      const cwd = process.env.GIT_PROJECT_PATH || "/Users/zvisegal/devlope/TaskYamGitMobile";
      const fileArg = file ? `-- ${file}` : "";
      let diff;
      try {
        diff = execSync(`git diff HEAD ${fileArg}`, { cwd, encoding: "utf-8", maxBuffer: 1024 * 1024 });
      } catch (e) {
        diff = "";
      }

      if (!diff.trim()) {
        return { content: [{ type: "text", text: "No uncommitted changes found." }] };
      }

      // Truncate if too large
      if (diff.length > 10000) {
        diff = diff.slice(0, 10000) + "\n\n... (truncated)";
      }

      let messages = [
        {
          role: "system",
          content:
            "You are an expert code reviewer. You will receive a git diff of code changes. " +
            "Analyze the changes and provide a clear report: what changed, potential bugs, regressions, " +
            "null safety issues, and any recommendations. Be concise and focus on real issues only. " +
            "You also have access to filesystem tools (list_directory, read_file, search_files, git_diff) " +
            "to read additional project files if needed for context. " +
            "The project is located at: '" + cwd + "'.",
        },
        { role: "user", content: `${query}\n\nHere is the git diff:\n\`\`\`\n${diff}\n\`\`\`` },
      ];

      let finalContent = null;
      const activeModel = await getLoadedModel(PREFERRED_MODEL);

      for (let step = 0; step < MAX_STEPS; step++) {
        const response = await axios.post(OLLAMA_URL, {
          model: activeModel,
          messages,
          temperature: 0.1,
          tools: FILESYSTEM_TOOLS,
        }, { timeout: LLM_TIMEOUT });

        const choice = response.data?.choices?.[0];
        const message = choice?.message;

        if (!message) {
          return {
            content: [{ type: "text", text: "Ollama returned no readable message." }],
            isError: true,
          };
        }

        messages.push(message);

        if (
          choice.finish_reason === "stop" ||
          (message.content && (!message.tool_calls || message.tool_calls.length === 0))
        ) {
          finalContent = message.content || "";
          break;
        }

        if (message.tool_calls && message.tool_calls.length > 0) {
          for (const toolCall of message.tool_calls) {
            const fnName = toolCall.function.name;
            let fnArgs;
            try {
              fnArgs = JSON.parse(toolCall.function.arguments);
            } catch {
              fnArgs = {};
            }

            console.error(`[git_review Step ${step + 1}] Tool call: ${fnName}(${JSON.stringify(fnArgs)})`);
            const result = executeToolCall(fnName, fnArgs);

            messages.push({
              role: "tool",
              tool_call_id: toolCall.id,
              content: result,
            });
          }
        }
      }

      if (finalContent === null) {
        return {
          content: [{ type: "text", text: "Ollama reached max steps without a final answer." }],
          isError: true,
        };
      }

      return { content: [{ type: "text", text: finalContent }] };
    } catch (error) {
      let errorMessage = "Unknown error communicating with Ollama.";
      if (axios.isAxiosError(error)) {
        errorMessage = `Ollama Error: ${error.response?.data?.error?.message || error.message}. Is Ollama running on port 11434?`;
      } else if (error instanceof Error) {
        errorMessage = `Internal Error: ${error.message}`;
      }
      return { content: [{ type: "text", text: errorMessage }], isError: true };
    }
  }

  if (request.params.name === "code_feedback") {
    const query = request.params.arguments?.query;
    const file = request.params.arguments?.file || "";

    if (!query || typeof query !== "string") {
      return {
        content: [{ type: "text", text: "Invalid arguments: 'query' is required and must be a string." }],
        isError: true,
      };
    }

    try {
      const cwd = process.env.GIT_PROJECT_PATH || "/Users/zvisegal/devlope/TaskYamGitMobile";
      const fileArg = file ? `-- ${file}` : "";
      let diff;
      try {
        diff = execSync(`git diff HEAD ${fileArg}`, { cwd, encoding: "utf-8", maxBuffer: 1024 * 1024 });
      } catch (e) {
        diff = "";
      }

      if (!diff.trim()) {
        return { content: [{ type: "text", text: "No uncommitted changes found." }] };
      }

      if (diff.length > 10000) {
        diff = diff.slice(0, 10000) + "\n\n... (truncated)";
      }

      const systemPrompt =
        "You are an expert code reviewer. You will receive a git diff of uncommitted changes. " +
        "Your job is to provide implementation feedback ONLY — do NOT suggest rewriting code. " +
        "Focus on:\n" +
        "- Code quality patterns and anti-patterns\n" +
        "- Modularity and reuse opportunities\n" +
        "- Readability and maintainability concerns\n" +
        "- Naming conventions consistency\n" +
        "- Potential edge cases or missing validations\n" +
        "Be concise. Comment only on real issues, not style nitpicks. " +
        "You also have access to filesystem tools (list_directory, read_file, search_files, git_diff) " +
        "to read additional project files if needed for context. " +
        "The project is located at: '" + cwd + "'.";

      const userMessage = `${query}\n\nHere is the git diff:\n\`\`\`\n${diff}\n\`\`\``;

      // ── Helper: run one model through the agentic loop ──
      async function runModel(modelId, label) {
        const messages = [
          { role: "system", content: systemPrompt },
          { role: "user", content: userMessage },
        ];
        for (let step = 0; step < MAX_STEPS; step++) {
          const response = await axios.post(OLLAMA_URL, {
            model: modelId,
            messages,
            temperature: 0.1,
            tools: FILESYSTEM_TOOLS,
          }, { timeout: LLM_TIMEOUT });

          const choice = response.data?.choices?.[0];
          const message = choice?.message;
          if (!message) return `[${label}] returned no readable message.`;

          messages.push(message);

          if (
            choice.finish_reason === "stop" ||
            (message.content && (!message.tool_calls || message.tool_calls.length === 0))
          ) {
            return message.content || "";
          }

          if (message.tool_calls?.length > 0) {
            for (const toolCall of message.tool_calls) {
              let fnArgs;
              try { fnArgs = JSON.parse(toolCall.function.arguments); } catch { fnArgs = {}; }
              console.error(`[${label} Step ${step + 1}] Tool call: ${toolCall.function.name}(${JSON.stringify(fnArgs)})`);
              messages.push({
                role: "tool",
                tool_call_id: toolCall.id,
                content: executeToolCall(toolCall.function.name, fnArgs),
              });
            }
          }
        }
        return `[${label}] reached max steps without a final answer.`;
      }

      // ── Run both models in parallel ──
      const [deepseekModel, coderModel] = await Promise.all([
        getLoadedModel(PREFERRED_DEEPSEEK),
        getLoadedModel(PREFERRED_MODEL),
      ]);

      const [deepseekResult, coderResult] = await Promise.all([
        runModel(deepseekModel, "DeepSeek-R1 (reasoning)"),
        runModel(coderModel, "Qwen3-Coder (patterns)"),
      ]);

      const combined =
        `## DeepSeek-R1 — Bugs & Edge Cases\n\n${deepseekResult}\n\n` +
        `---\n\n## Qwen3-Coder — Patterns & Code Quality\n\n${coderResult}`;

      return { content: [{ type: "text", text: combined }] };
    } catch (error) {
      let errorMessage = "Unknown error communicating with Ollama.";
      if (axios.isAxiosError(error)) {
        errorMessage = `Ollama Error: ${error.response?.data?.error?.message || error.message}. Is Ollama running on port 11434?`;
      } else if (error instanceof Error) {
        errorMessage = `Internal Error: ${error.message}`;
      }
      return { content: [{ type: "text", text: errorMessage }], isError: true };
    }
  }

  throw new Error(`Tool not found: ${request.params.name}`);
});

// 4. Start the server
async function run() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Ollama Docs, Git Review & Code Feedback MCP Server v1.4 running.");
}

run().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
