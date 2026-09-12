import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../", import.meta.url));
const frontend = fileURLToPath(new URL("../", import.meta.url));
const python = process.env.REZX_PYTHON || `${root}.venv/bin/python`;
const apiPort = Number(process.env.REZX_API_PORT || 8000);
const frontendPort = Number(process.env.REZX_FRONTEND_PORT || 5173);
const apiURL = `http://127.0.0.1:${apiPort}`;
const children = [];
let stopping = false;

function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach((child) => child.kill("SIGTERM"));
  process.exitCode = code;
}

function launch(command, args, cwd, env = process.env) {
  const child = spawn(command, args, { cwd, env, stdio: "inherit" });
  children.push(child);
  child.on("error", (error) => {
    console.error(error.message);
    stop(1);
  });
  child.on("exit", (code) => {
    if (!stopping) stop(code || 0);
  });
  return child;
}

async function health() {
  try {
    const response = await fetch(`${apiURL}/api/health`, {
      signal: AbortSignal.timeout(1000),
    });
    return await response.json();
  } catch {
    return null;
  }
}

process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());

if (
  ![apiPort, frontendPort].every(
    (port) => Number.isInteger(port) && port > 0 && port < 65536,
  )
) {
  throw new Error(
    "REZX_API_PORT and REZX_FRONTEND_PORT must be valid port numbers.",
  );
}
let status = await health();
if (!status) {
  if (!existsSync(python))
    throw new Error(
      "Python environment missing. Follow web_api/README.md or set REZX_PYTHON.",
    );
  launch(
    python,
    [
      "-m",
      "uvicorn",
      "web_api.app:app",
      "--host",
      "127.0.0.1",
      "--port",
      String(apiPort),
    ],
    root,
  );
  for (let attempt = 0; attempt < 60 && !stopping; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 500));
    status = await health();
    if (status) break;
  }
}
if (!stopping) {
  if (status?.model !== "S2-EvidenceSR-4X" || status.status !== "ready") {
    console.error(
      status?.error ||
        "The model API did not become ready. Check the checkpoint and backend log.",
    );
    stop(1);
  } else {
    console.log(
      `RezX model connected: step ${status.step} on ${status.device}.`,
    );
    launch(
      process.execPath,
      [
        "node_modules/vite/bin/vite.js",
        "--host",
        "0.0.0.0",
        "--strictPort",
        "--port",
        String(frontendPort),
      ],
      frontend,
      { ...process.env, REZX_API_URL: apiURL },
    );
  }
}
