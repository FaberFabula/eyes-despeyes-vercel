import { HardLearning } from "./hard-learning.js";
import { createFramePacer } from "./frame-pacing.js";

const runtimeURL = "https://cdn.jsdelivr.net/pyodide/v314.0.8/full/";
const pymunkWheel = new URL(
  "./wheels/pymunk-7.3.1-cp314-cp314-pyemscripten_2026_0_wasm32.whl",
  import.meta.url,
).href;
const gameAssets = [
  "assets/mobs.png", "assets/items.png", "assets/tiles.png", "assets/epics.png", "assets/debug.png", "assets/sprites.json",
];
const sourceDirectory = "/game/maze_game";

const canvas = document.querySelector("#canvas");
const stage = document.querySelector(".stage");
const launcher = document.querySelector("#launcher");
const playButtons = [...launcher.querySelectorAll("button[data-mode]")];

const status = document.querySelector("#status");
const errorDetails = document.querySelector("#draft-error");
const editorLink = document.querySelector("#back-to-studio");
const draftMode = editorLink !== null && new URLSearchParams(location.search).get("draft") === "1";
if (editorLink !== null) {
  editorLink.hidden = !draftMode;
  document.querySelector("#draft-notice").hidden = !draftMode;
  document.querySelector("#play-baseline").hidden = !draftMode;
  document.querySelector(".hard-mode-action").hidden = draftMode;
}
let engine;
let loadedSources;
let focusOnNextFrame = false;
let pointer = [0.5, 0.5];
let viewport = [828, 700];
let inGame = false;
let inFullscreen = false;
let pauseOnFullscreenExit = false;
const frames = createFramePacer();

const soundBuffers = new Map();
let gameAudio;

function soundURL(kind) {
  return new URL(`./maze_game/assets/audio/${kind}.wav`, import.meta.url);
}

function enableGameAudio() {
  const Audio = window.AudioContext ?? window.webkitAudioContext;
  if (!Audio) return;
  gameAudio ??= new Audio();
  if (gameAudio.state !== "running") void gameAudio.resume().catch(() => {});
}

async function playGameCue(kind, volume, pan) {
  if (gameAudio?.state !== "running") return;
  if (!soundBuffers.has(kind)) {
    const buffer = fetch(soundURL(kind))
      .then((response) => {
        if (!response.ok) throw new Error(`Missing game sound ${kind}`);
        return response.arrayBuffer();
      })
      .then((bytes) => gameAudio.decodeAudioData(bytes));
    soundBuffers.set(kind, buffer);
  }
  const buffer = await soundBuffers.get(kind).catch(() => {
    soundBuffers.delete(kind);
    return null;
  });
  if (!buffer || gameAudio.state !== "running") return;

  const sample = gameAudio.createBufferSource();
  sample.buffer = buffer;
  const gain = gameAudio.createGain();
  gain.gain.value = volume;
  sample.connect(gain);
  if (gameAudio.createStereoPanner) {
    const stereo = gameAudio.createStereoPanner();
    stereo.pan.value = pan;
    gain.connect(stereo).connect(gameAudio.destination);
  } else {
    gain.connect(gameAudio.destination);
  }
  sample.start();
}

function showStatus(message) {
  status.textContent = message;
  status.hidden = !message;
}

async function fetchSource(file) {
  const response = await fetch("./maze_game/" + file, { cache: "no-cache" });
  if (!response.ok) throw new Error(`Loading ${file} returned ${response.status}`);
  return response;
}

async function loadEngine() {
  showStatus("Loading the game engine…");
  const { loadPyodide } = await import(`${runtimeURL}pyodide.mjs`);
  const python = await loadPyodide({ indexURL: runtimeURL });
  python.canvas.setCanvas2D(canvas);

  showStatus("Loading physics and sprites…");
  python.FS.mkdirTree(`${sourceDirectory}/assets`);
  await Promise.all([
    python.loadPackage(["pygame-ce", "cffi", "shapely", pymunkWheel]),
    ...gameAssets.map(async (file) => {
      const response = await fetchSource(file);
      python.FS.writeFile(`${sourceDirectory}/${file}`, new Uint8Array(await response.arrayBuffer()));
    }),
  ]);
  python.FS.chdir("/game");
  python.globals.set("game_pointer", () => [pointer[0] * canvas.width, pointer[1] * canvas.height]);
  python.globals.set("game_size", () => viewport);
  python.globals.set("game_pause", () => {
    const pause = pauseOnFullscreenExit;
    pauseOnFullscreenExit = false;
    return pause;
  });
  python.globals.set("next_frame", async () => {
    await frames.next();
    if (focusOnNextFrame) {
      canvas.focus({ preventScroll: true });
      focusOnNextFrame = false;
    }
  });
  python.globals.set("play_sound", (kind, volume, pan) => {
    void playGameCue(kind, volume, pan);
  });
  return python;
}

async function loadGameCode(python, drafts) {
  showStatus("Loading Python files…");
  const manifest = await fetch("./release-files.json");
  if (!manifest.ok) throw new Error("The game file list could not be loaded.");
  const { pythonFiles } = await manifest.json();
  const sources = await Promise.all(pythonFiles.map(async (file) => [
    file, drafts.get(file) ?? await (await fetchSource(file)).text(),
  ]));
  if (loadedSources && sources.every(([file, code]) => loadedSources.get(file) === code)) return;
  loadedSources = undefined;
  for (const [file, code] of sources) python.FS.writeFile(`${sourceDirectory}/${file}`, code);
  // Keep the runtime and assets warm, and reimport the game when its code changes.
  python.runPython(`
import importlib
import sys

sys.dont_write_bytecode = True
for module in tuple(sys.modules):
    if module == "maze_game" or module.startswith("maze_game."):
        del sys.modules[module]
importlib.invalidate_caches()
from maze_game.main import run as play_maze
`);
  loadedSources = new Map(sources);
}

canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => event.preventDefault(), { passive: false });
function trackPointer(event) {
  const rect = canvas.getBoundingClientRect();
  pointer = [
    (event.clientX - rect.left) / rect.width,
    (event.clientY - rect.top) / rect.height,
  ];
}
function resizeGame() {
  const { width, height } = stage.getBoundingClientRect();
  const scale = Math.max(700 / height, 680 / width);
  viewport = [Math.round(width * scale), Math.round(height * scale)];
}
window.addEventListener("resize", resizeGame);
document.addEventListener("fullscreenchange", () => {
  const full = document.fullscreenElement === stage;
  if (inGame && inFullscreen && !full) pauseOnFullscreenExit = true;
  inFullscreen = full;
  resizeGame();
});
window.addEventListener("pointermove", trackPointer);
canvas.addEventListener("pointerdown", (event) => {
  trackPointer(event);
  canvas.focus({ preventScroll: true });
  enableGameAudio();
});
canvas.addEventListener("keydown", (event) => {
  if (event.code === "Space") event.preventDefault();
});


function readDrafts() {
  if (!draftMode) return new Map();
  const saved = JSON.parse(localStorage.getItem("i-despeyes-you-draft-v1")) ?? {};
  return new Map(Object.entries(saved)
    .filter(([path, code]) => path.startsWith("maze_game/") && typeof code === "string")
    .map(([path, code]) => [path.slice("maze_game/".length), code]));
}

async function startGame(button) {
  const sandbox = button.dataset.mode === "sandbox";
  const hard = button.dataset.mode === "hard" ? new HardLearning() : null;
  errorDetails.hidden = true;
  enableGameAudio();
  inGame = true;
  pauseOnFullscreenExit = false;
  if (!document.fullscreenElement) void stage.requestFullscreen?.()?.catch(() => {});
  resizeGame();
  for (const entry of playButtons) entry.disabled = true;
  try {
    const learning = hard?.open();
    engine ??= await loadEngine();
    await loadGameCode(engine, readDrafts());
    await learning;
    showStatus("");
    launcher.hidden = true;
    canvas.focus({ preventScroll: true });
    focusOnNextFrame = true;
    frames.reset();
    const options = "play_sound=play_sound, browser_pointer=game_pointer, browser_size=game_size, browser_pause=game_pause";
    let mode = `sandbox=${sandbox ? "True" : "False"}`;
    if (hard) {
      engine.globals.set("hard_prior", () => hard.prior());
      engine.globals.set("hard_report", (event, value) => hard.receive(event, value));
      mode = "hard_mode=True, learning_prior=hard_prior, report_learning=hard_report";
    }
    await engine.runPythonAsync(`await play_maze(next_frame, ${mode}, ${options})`);
  } catch (error) {
    errorDetails.hidden = false;
    document.querySelector("#draft-traceback").textContent = String(error);
    showStatus("The game stopped. Open the error details to see where.");
  } finally {
    hard?.close();
    inGame = false;
    if (document.fullscreenElement === stage) await document.exitFullscreen().catch(() => {});
    launcher.hidden = false;
    for (const entry of playButtons) entry.disabled = false;
    button.focus({ preventScroll: true });
  }
}

for (const button of playButtons) {
  button.addEventListener("click", () => startGame(button));
}
