const poolUpdate = `
local checkpoint = ARGV[1]
if redis.call('HEXISTS', KEYS[1], checkpoint) ~= 0 then
  return 0
end
for index = 3, #ARGV, 3 do
  local count_key = 'n:' .. ARGV[index]
  local sum_key = 's:' .. ARGV[index]
  local count = tonumber(redis.call('HGET', KEYS[1], count_key) or '0')
  local total = tonumber(redis.call('HGET', KEYS[1], sum_key) or '0')
  local next_count = redis.call('HINCRBY', KEYS[1], count_key, ARGV[index + 1])
  local next_total = redis.call('HINCRBY', KEYS[1], sum_key, ARGV[index + 2])
  local weight = math.min(48, count)
  local next_weight = math.min(48, next_count)
  local prior_reward = count == 0 and 0 or math.floor(total * weight / count + 0.5)
  local next_reward = math.floor(next_total * next_weight / next_count + 0.5)
  redis.call('HINCRBY', KEYS[2], count_key, next_weight - weight)
  redis.call('HINCRBY', KEYS[2], sum_key, next_reward - prior_reward)
end
redis.call('HSET', KEYS[1], checkpoint, ARGV[2])
return 1
`;

export function learningStore(env = process.env) {
  const url = env.UPSTASH_REDIS_REST_URL || env.KV_REST_API_URL;
  const token = env.UPSTASH_REDIS_REST_TOKEN || env.KV_REST_API_TOKEN;
  if (!url || !token) return null;
  const namespace = "eyes-despeyes:hard:v1:" + (env.VERCEL_ENV === "preview" ? "preview:" : "");
  const poolKey = namespace + "pool";

  async function command(args) {
    const response = await fetch(url, {
      method: "POST",
      headers: { Authorization: "Bearer " + token, "Content-Type": "application/json" },
      body: JSON.stringify(args),
      signal: AbortSignal.timeout(3000),
    });
    if (!response.ok) throw new Error("Learning store unavailable");
    const { result, error } = await response.json();
    if (error) throw new Error("Learning store rejected the command");
    return result;
  }

  return {
    async prior() {
      const fields = await command(["HGETALL", poolKey]);
      const values = new Map();
      if (Array.isArray(fields)) {
        for (let i = 0; i < fields.length; i += 2) values.set(fields[i], Number(fields[i + 1]));
      } else if (fields && typeof fields === "object") {
        for (const [key, value] of Object.entries(fields)) values.set(key, Number(value));
      }
      const rows = [];
      for (const [key, count] of values) {
        if (key.startsWith("n:") && count > 0) {
          const identity = key.slice(2);
          rows.push([...JSON.parse(identity), count, values.get("s:" + identity) ?? 0]);
        }
      }
      return rows;
    },

    async save(batch) {
      const samples = batch.rows.flatMap(([task, context, action, count, total]) =>
        [JSON.stringify([task, context, action]), count, total]);
      const saved = await command([
        "EVAL", poolUpdate, 2, namespace + "run:" + batch.run, poolKey,
        "checkpoint:" + batch.checkpoint, JSON.stringify(batch), ...samples,
      ]);
      return Boolean(saved);
    },
  };
}
