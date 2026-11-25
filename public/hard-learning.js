const endpoint = "/api/hard-learning";
const keyPrefix = "eyes-despeyes-hard-v1:";
const poolKey = keyPrefix + "pool";
const checkpointInterval = 10_000;
const terminal = new Set(["defeat", "restart", "leave"]);

function readSaved(key) {
  try {
    return JSON.parse(localStorage.getItem(key));
  } catch {
    return null;
  }
}

function writeSaved(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // Keep the pending checkpoint in memory when browser storage is full or blocked.
  }
}

function addRows(pool, rows) {
  for (const [task, context, action, count, total] of rows) {
    const key = JSON.stringify([task, context, action]);
    const row = pool.get(key) ?? [task, context, action, 0, 0];
    row[3] += count;
    row[4] += total;
    pool.set(key, row);
  }
}

function savedCheckpoints() {
  const records = new Map();
  try {
    for (let index = 0; index < localStorage.length; index++) {
      const key = localStorage.key(index);
      if (!key?.startsWith(keyPrefix) || key === poolKey) continue;
      const record = readSaved(key);
      if (Array.isArray(record?.body?.rows)) records.set(key, record);
    }
  } catch {
    // The run can learn in memory even when local storage is disabled.
  }
  return records;
}

export class HardLearning {
  constructor() {
    this.records = savedCheckpoints();
    this.pool = new Map();
    this.run = null;
    this.cloud = false;
    this.sending = false;
    this.hidden = () => {
      if (document.visibilityState === "hidden") this.checkpoint("progress", true);
    };
    this.depart = (event) => this.checkpoint(event.persisted ? "progress" : "leave", true);
    this.reconnect = () => void this.enableCloud();
  }

  async open() {
    const cachedPool = readSaved(poolKey);
    const included = new Set(Array.isArray(cachedPool?.included) ? cachedPool.included : []);
    if (Array.isArray(cachedPool?.rows)) addRows(this.pool, cachedPool.rows);
    for (const [key, { body }] of this.records) {
      if (!included.has(key)) addRows(this.pool, body.rows);
    }
    await this.enableCloud(true);
    if (this.closed) return this;
    this.timer = setInterval(() => {
      this.checkpoint("progress");
      void this.upload();
    }, checkpointInterval);
    document.addEventListener("visibilitychange", this.hidden);
    window.addEventListener("pagehide", this.depart);
    window.addEventListener("online", this.reconnect);
    return this;
  }

  async enableCloud(replacePool = false) {
    try {
      const included = [...this.records].filter(([, record]) => record.sent).map(([key]) => key);
      const response = await fetch(endpoint, { cache: "no-store", signal: AbortSignal.timeout(2200) });
      if (!response.ok) return;
      const { prior } = await response.json();
      if (!Array.isArray(prior)) return;
      this.cloud = true;
      writeSaved(poolKey, { rows: prior, included });
      if (replacePool) {
        this.pool.clear();
        addRows(this.pool, prior);
        for (const { body, sent } of this.records.values()) {
          if (!sent) addRows(this.pool, body.rows);
        }
      }
      void this.upload();
    } catch {
      // Keep recording locally. A later online run can upload the pending checkpoints.
    }
  }

  prior() {
    return JSON.stringify([...this.pool.values()]);
  }

  receive(event, value) {
    if (event === "start") {
      this.checkpoint("restart");
      this.run = { id: crypto.randomUUID(), number: 0, floor: Number(value), rows: new Map(), count: 0 };
      this.checkpoint("start");
      return;
    }
    if (!this.run) return;
    if (event === "sample") {
      const [task, context, action, reward] = JSON.parse(value);
      addRows(this.run.rows, [[task, context, action, 1, reward]]);
      this.run.count++;
      if (this.run.rows.size >= 150 || this.run.count >= 1000) this.checkpoint("progress");
      return;
    }
    this.run.floor = Number(value);
    this.checkpoint(event);
  }

  checkpoint(event, keepalive = false) {
    const run = this.run;
    if (!run || (event === "progress" && !run.count)) return;
    const body = {
      version: 1, run: run.id, checkpoint: run.number++, floor: run.floor, event,
      rows: [...run.rows.values()],
    };
    run.rows.clear();
    run.count = 0;
    if (terminal.has(event)) this.run = null;
    addRows(this.pool, body.rows);
    const key = `${keyPrefix}${body.run}:${body.checkpoint}`;
    const record = { body, sent: false };
    this.records.set(key, record);
    writeSaved(key, record);
    if (this.cloud && keepalive) {
      void this.send(key, record, true);
    } else {
      void this.upload();
    }
  }

  async send(key, record, keepalive = false) {
    try {
      const response = await fetch(endpoint, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record.body), keepalive, signal: AbortSignal.timeout(4500),
      });
      if (!response.ok) return false;
      record.sent = true;
      writeSaved(key, record);
      return true;
    } catch {
      return false;
    }
  }

  async upload() {
    if (!this.cloud || this.sending) return;
    this.sending = true;
    try {
      for (const [key, record] of this.records) {
        if (!record.sent && !await this.send(key, record)) break;
      }
    } finally {
      this.sending = false;
    }
  }

  close() {
    this.closed = true;
    this.checkpoint("leave");
    clearInterval(this.timer);
    document.removeEventListener("visibilitychange", this.hidden);
    window.removeEventListener("pagehide", this.depart);
    window.removeEventListener("online", this.reconnect);
  }
}
