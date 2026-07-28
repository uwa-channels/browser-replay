#!/usr/bin/env node
// Builds core/ to wasm32 and generates JS bindings into web/src/wasm/.
//
// Calls `cargo build` + `wasm-bindgen` directly rather than `wasm-pack`:
// wasm-pack insists on installing its own copy of the wasm-bindgen CLI
// (matching the crate version exactly) into its own cache on every run,
// which in this environment repeatedly hung fetching that binary. Calling
// the already-installed `wasm-bindgen` CLI directly sidesteps that -- it
// just requires the CLI version to match the `wasm-bindgen` crate version
// pinned in core/Cargo.toml exactly (currently 0.2.125).

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import os from "node:os";
import path from "node:path";

const webDir = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const coreDir = path.join(webDir, "..", "core");
const outDir = path.join(webDir, "src", "wasm");
const profile = process.env.WASM_PROFILE === "dev" ? "debug" : "release";

function run(cmd, args, cwd) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { cwd, stdio: "inherit" });
}

// `~/.cargo/bin` isn't always on PATH for non-login shells (observed in this
// project's dev environment); fall back to it explicitly if the bare command
// isn't resolvable.
function resolveBin(name) {
  try {
    execFileSync(name, ["--version"], { stdio: "ignore" });
    return name;
  } catch {
    const fallback = path.join(os.homedir(), ".cargo", "bin", name);
    if (existsSync(fallback)) return fallback;
    throw new Error(`${name} not found on PATH or in ~/.cargo/bin`);
  }
}

const cargoArgs = ["build", "--target", "wasm32-unknown-unknown"];
if (profile === "release") cargoArgs.push("--release");
run(resolveBin("cargo"), cargoArgs, coreDir);

if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

const wasmPath = path.join(coreDir, "target", "wasm32-unknown-unknown", profile, "replay_core.wasm");
run(resolveBin("wasm-bindgen"), ["--target", "web", "--out-dir", outDir, wasmPath], webDir);

console.log(`WASM bindings written to ${outDir}`);
