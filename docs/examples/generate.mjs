#!/usr/bin/env node
// DiffusionBear from Node 18+ (built-in fetch, no dependencies).
//   DIFFUSIONBEAR_URL=http://127.0.0.1:8001 DIFFUSIONBEAR_API_KEY=dbk_... \
//     node docs/examples/generate.mjs "an origami fox, studio light" fox.png
import { writeFile } from "node:fs/promises";
import { basename } from "node:path";

const BASE = (process.env.DIFFUSIONBEAR_URL ?? "http://127.0.0.1:8001").replace(/\/$/, "");
const KEY = process.env.DIFFUSIONBEAR_API_KEY ?? "";
const [prompt = "a small red apple on a white table", out = "out.png"] = process.argv.slice(2);
const headers = KEY ? { Authorization: `Bearer ${KEY}` } : {};

async function api(path, init = {}) {
  const res = await fetch(BASE + path, { ...init, headers: { ...headers, ...init.headers } });
  if (!res.ok) {
    const retry = res.headers.get("retry-after");
    throw new Error(`${path} -> HTTP ${res.status}: ${await res.text()}${retry ? ` (retry after ${retry}s)` : ""}`);
  }
  return res;
}
const json = async (path, init) => (await api(path, init)).json();

const models = await json("/api/models");
const model = process.env.MODEL ?? models.find((m) => m.installed)?.id;
if (!model) throw new Error("No installed model — download one in the Models tab.");

const { job_id } = await json("/api/generate", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ model, prompt, width: 512, height: 768 }),
});
process.on("SIGINT", async () => {
  await api(`/api/jobs/${job_id}/cancel`, { method: "POST" }).catch(() => {});
  process.exit(130);
});

let job;
do {
  await new Promise((r) => setTimeout(r, 1500));
  job = await json(`/api/jobs/${job_id}`);
  const p = job.progress ?? {};
  process.stderr.write(`\r${job.status} ${p.step ?? "-"}/${p.steps ?? "-"}   `);
} while (!["done", "error", "cancelled"].includes(job.status));
process.stderr.write("\n");
if (job.status !== "done") throw new Error(`job ${job.status}: ${job.error}`);

const id = job.results[0].id;
await writeFile(out, Buffer.from(await (await api(`/api/images/${id}/file`)).arrayBuffer()));
const sidecar = /\.\w+$/.test(basename(out)) ? out.replace(/\.\w+$/, ".json") : `${out}.json`;
await writeFile(sidecar, JSON.stringify(await json(`/api/images/${id}`), null, 2));
console.log(`saved ${out} (seed ${job.results[0].seed})`);
