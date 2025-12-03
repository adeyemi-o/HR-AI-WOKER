const MODELS = {
  chat: "@cf/meta/llama-3-8b-instruct",
  reasoning: "@cf/deepseek/deepseek-r1-distill-qwen-32b",
  embedding: "@cf/baai/bge-large-en-v1.5",
};

const ALLOWED_TASKS = new Set(["chat", "reasoning", "embedding"]);
const MAX_MESSAGES = 64;
const MAX_REQUEST_BYTES = 256 * 1024; // 256 KiB guardrail to prevent abuse

const DEFAULT_CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, x-api-key",
  "Access-Control-Max-Age": "86400",
};

const JSON_HEADERS = {
  "Content-Type": "application/json",
};

const TASK_ALIASES = {
  embed: "embedding",
  embeddings: "embedding",
  vectorize: "embedding",
  reason: "reasoning",
  analysis: "reasoning",
  chat: "chat",
  conversation: "chat",
};

/**
 * Normalize incoming task names to a supported canonical task.
 * Returns null for unsupported tasks to prevent unexpected model routing.
 */
function normalizeTask(task) {
  if (!task) return "chat";
  const cleaned = String(task).trim().toLowerCase();
  const normalized = TASK_ALIASES[cleaned] || cleaned;
  return ALLOWED_TASKS.has(normalized) ? normalized : null;
}

function withCors(body, init = {}) {
  const headers = { ...DEFAULT_CORS_HEADERS, ...(init.headers || {}) };
  return new Response(body, { ...init, headers });
}

function errorResponse(status, message, code) {
  return withCors(
    JSON.stringify({ success: false, error: { code, message } }),
    {
      status,
      headers: JSON_HEADERS,
    },
  );
}

function validateApiKey(request, env) {
  const providedKey = request.headers.get("x-api-key");
  if (!providedKey || !env.API_KEY || providedKey.trim() !== env.API_KEY) {
    return false;
  }
  return true;
}

function validateContentType(request) {
  const contentType = request.headers.get("content-type") || "";
  return contentType.toLowerCase().includes("application/json");
}

function ensureMessages(input) {
  if (!input || !Array.isArray(input.messages) || input.messages.length === 0) {
    return false;
  }
  if (input.messages.length > MAX_MESSAGES) {
    return false;
  }
  return input.messages.every(
    (msg) =>
      msg &&
      typeof msg.role === "string" &&
      msg.role.trim().length > 0 &&
      typeof msg.content === "string" &&
      msg.content.trim().length > 0,
  );
}

function ensureEmbeddingInput(input) {
  if (!input || typeof input.text !== "string") return false;
  const trimmed = input.text.trim();
  return trimmed.length > 0 && trimmed.length <= 8000;
}

async function runChatModel(env, model, input) {
  const payload = {
    messages: input.messages,
    temperature:
      typeof input.temperature === "number"
        ? Math.min(Math.max(input.temperature, 0), 2)
        : 0.7,
    max_tokens:
      typeof input.max_tokens === "number"
        ? Math.max(Math.min(input.max_tokens, 2048), 1)
        : 512,
  };

  return env.AI.run(model, payload);
}

async function runEmbeddingModel(env, model, input) {
  return env.AI.run(model, { text: input.text.trim() });
}

function selectModel(task) {
  switch (task) {
    case "embedding":
      return MODELS.embedding;
    case "reasoning":
      return MODELS.reasoning;
    case "chat":
    default:
      return MODELS.chat;
  }
}

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return withCors(null, { status: 204 });
    }

    if (!env?.AI) {
      return errorResponse(500, "AI binding is not configured.", "misconfigured_worker");
    }

    if (!env?.API_KEY) {
      return errorResponse(500, "API key is not configured.", "missing_api_key_secret");
    }

    if (request.method !== "POST") {
      return errorResponse(405, "Method not allowed. Use POST.", "method_not_allowed");
    }

    if (!validateApiKey(request, env)) {
      return errorResponse(401, "Invalid or missing API key.", "unauthorized");
    }

    if (!validateContentType(request)) {
      return errorResponse(415, "Content-Type must be application/json.", "invalid_content_type");
    }

    const contentLengthHeader = request.headers.get("content-length");
    const declaredLength = contentLengthHeader ? Number(contentLengthHeader) : 0;
    if (declaredLength && declaredLength > MAX_REQUEST_BYTES) {
      return errorResponse(413, "Payload too large.", "payload_too_large");
    }

    let payload;
    try {
      const rawBody = await request.text();
      if (!declaredLength && rawBody.length > MAX_REQUEST_BYTES) {
        return errorResponse(413, "Payload too large.", "payload_too_large");
      }
      payload = JSON.parse(rawBody || "{}");
    } catch (err) {
      return errorResponse(400, "Invalid JSON payload.", "invalid_json");
    }

    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      return errorResponse(400, "Request body must be a JSON object.", "invalid_body_type");
    }

    const task = normalizeTask(payload.task);
    if (!task) {
      return errorResponse(400, "Unsupported task type.", "invalid_task");
    }

    const input = payload.input;

    if (!input || typeof input !== "object") {
      return errorResponse(400, "Input payload is required.", "missing_input");
    }

    const model = selectModel(task);

    try {
      let aiResult;
      if (task === "embedding") {
        if (!ensureEmbeddingInput(input)) {
          return errorResponse(400, "Embedding requests require a non-empty text field.", "invalid_embedding_input");
        }
        aiResult = await runEmbeddingModel(env, model, input);
      } else {
        if (!ensureMessages(input)) {
          return errorResponse(400, "Chat requests require a messages array with role and content.", "invalid_chat_input");
        }
        aiResult = await runChatModel(env, model, input);
      }

      const responseBody = {
        success: true,
        task,
        model,
        result: aiResult,
      };

      return withCors(JSON.stringify(responseBody), { status: 200, headers: JSON_HEADERS });
    } catch (err) {
      return errorResponse(502, "AI provider error. Please retry.", "ai_inference_failed");
    }
  },
};
