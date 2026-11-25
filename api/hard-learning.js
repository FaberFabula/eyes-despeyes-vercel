import { learningStore } from "../lib/learning-store.js";

const maxBytes = 48 * 1024;
const events = new Set(["start", "progress", "floor_start", "floor_clear", "defeat", "restart", "leave"]);
const weapons = [
  "Sword", "Hammer", "Slingshot", "Flamethrower", "Gravity Gun", "Shield", "Bow", "Wand", "Laser",
  "Radar", "Spear", "Man Catcher", "Pulse Gun", "Chaotic Grimoire", "Sealing Equipment",
  "Cat-o'-nine-tails", "All-launcher",
];
const modes = new Set(["empty", "innate", "unknown", "melee", "ranged", "magic", "shield", "utility"]);
const aimActions = new Set(["direct", "lead", "counter_dodge"]);
const taskActions = new Map([
  ["combat", new Set(["close", "left", "right", "back", "pursue", "attack", "search_weapon", "contest",
    "secure_key", ...weapons.map((name) => "take:" + name)])],
  ["weapon", new Set(weapons)],
  ["dodge", new Set(["hold", "left", "right"])],
  ["patrol", new Set(["patrol", "search", "guard"])],
  ["aim_melee", aimActions],
  ["aim_ranged", aimActions],
  ["aim_magic", aimActions],
  ["mimic_attack", new Set(["impale", "swing", "bash"])],
  ["mimic_hunt", modes],
]);

function validContext(task, context) {
  if (!Array.isArray(context)) return false;
  const [first, second, third] = context;
  if (task === "mimic_hunt") return context.length === 1 && ["small", "large"].includes(first);
  if (task === "mimic_attack") {
    return context.length === 2 && modes.has(first) && ["close", "reach"].includes(second);
  }
  if (task === "patrol") {
    return context.length === 2 && ["trail", "quiet"].includes(first) && ["armed", "empty"].includes(second);
  }
  if (task === "weapon") {
    return context.length === 2 && modes.has(first) && ["near", "far"].includes(second);
  }
  return context.length === 3 && modes.has(first) && modes.has(second) && ["near", "far"].includes(third);
}

function checkpoint(data) {
  if (data?.version !== 1 || typeof data.run !== "string" ||
      !/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(data.run) ||
      !Number.isInteger(data.checkpoint) || data.checkpoint < 0 || data.checkpoint > 100_000 ||
      !Number.isInteger(data.floor) || data.floor < 1 || data.floor > 10_000 || !events.has(data.event) ||
      !Array.isArray(data.rows) || data.rows.length > 160) return null;

  const keys = new Set();
  let samples = 0;
  for (const row of data.rows) {
    if (!Array.isArray(row) || row.length !== 5) return null;
    const [task, context, action, count, total] = row;
    if (!taskActions.get(task)?.has(action) || !validContext(task, context) ||
        !Number.isInteger(count) || count < 1 || count > 1000 ||
        !Number.isInteger(total) || Math.abs(total) > 1000 * count) return null;
    const key = JSON.stringify([task, context, action]);
    if (keys.has(key)) return null;
    keys.add(key);
    samples += count;
  }
  if (samples > 1000 || (data.event === "progress" && !samples)) return null;
  return { version: 1, run: data.run, checkpoint: data.checkpoint, floor: data.floor,
    event: data.event, rows: data.rows };
}

async function readCheckpoint(request) {
  if (!request.headers.get("content-type")?.startsWith("application/json") || !request.body) return null;
  const reader = request.body.getReader();
  const decoder = new TextDecoder();
  let size = 0;
  let text = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.byteLength;
    if (size > maxBytes) {
      await reader.cancel();
      return null;
    }
    text += decoder.decode(value, { stream: true });
  }
  return checkpoint(JSON.parse(text + decoder.decode()));
}

export async function handleLearning(request, store = learningStore()) {
  const respond = (value, status = 200) => Response.json(value, {
    status, headers: { "Cache-Control": "no-store" },
  });
  if (!["GET", "POST"].includes(request.method)) return respond({ error: "Method not allowed" }, 405);
  const origin = request.headers.get("origin");
  if (origin && origin !== new URL(request.url).origin) return respond({ error: "Origin not allowed" }, 403);
  if (!store) return respond({ error: "Shared learning is unavailable" }, 503);
  let batch;
  if (request.method === "POST") {
    try {
      batch = await readCheckpoint(request);
    } catch {
      batch = null;
    }
    if (!batch) return respond({ error: "Invalid learning checkpoint" }, 400);
  }
  try {
    if (request.method === "GET") return respond({ prior: await store.prior() });
    return respond({ saved: await store.save(batch) });
  } catch {
    return respond({ error: "Shared learning is unavailable" }, 503);
  }
}

export default {
  fetch(request) {
    return handleLearning(request);
  },
};
