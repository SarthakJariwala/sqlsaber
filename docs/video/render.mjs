#!/usr/bin/env node
/**
 * Render the SQLsaber feature video frame by frame and encode it for the web.
 *
 * composition.html exposes window.__seek(t); this script steps through every
 * frame in headless Chromium, saves lossless PNGs, and encodes them with
 * ffmpeg into docs/public/video/.
 *
 * Usage:
 *   node render.mjs                     # full render: AV1 + H.264 MP4s and a poster
 *   node render.mjs --stills 2,9.5,18   # PNG stills (seconds) for review
 *   node render.mjs --scale 1           # faster, no supersampling
 *
 * Environment:
 *   FFMPEG         ffmpeg binary (default: ffmpeg on PATH)
 *   CHROMIUM_PATH  Chromium binary (default: Playwright's managed Chromium)
 */
import { spawn } from "node:child_process";
import { createReadStream, existsSync, mkdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright-core";

const here = path.dirname(fileURLToPath(import.meta.url));
const docsRoot = path.resolve(here, "..");

const { values: opts } = parseArgs({
	options: {
		stills: { type: "string" },
		scale: { type: "string", default: "2" },
		workers: { type: "string", default: String(Math.max(1, Math.min(4, os.cpus().length))) },
		out: { type: "string", default: path.join(docsRoot, "public", "video") },
		frames: { type: "string", default: path.join(os.tmpdir(), "sqlsaber-video-frames") },
		name: { type: "string", default: "sqlsaber-feature" },
		poster: { type: "string", default: "17.9" },
		"keep-frames": { type: "boolean", default: false },
		"skip-capture": { type: "boolean", default: false },
	},
});

const FFMPEG = process.env.FFMPEG || "ffmpeg";
const scale = Number(opts.scale);

const MIME = {
	".html": "text/html; charset=utf-8",
	".js": "text/javascript; charset=utf-8",
	".css": "text/css; charset=utf-8",
	".woff2": "font/woff2",
	".woff": "font/woff",
};

function serve(root) {
	const server = http.createServer((req, res) => {
		const url = new URL(req.url ?? "/", "http://localhost");
		const file = path.join(root, decodeURIComponent(url.pathname));
		if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) {
			res.writeHead(404).end();
			return;
		}
		res.writeHead(200, { "content-type": MIME[path.extname(file)] ?? "application/octet-stream" });
		createReadStream(file).pipe(res);
	});
	return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

function run(cmd, args) {
	return new Promise((resolve, reject) => {
		const child = spawn(cmd, args, { stdio: ["ignore", "inherit", "inherit"] });
		child.on("error", reject);
		child.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`${cmd} exited with ${code}`))));
	});
}

async function openPage(browser, url) {
	const context = await browser.newContext({
		viewport: { width: 1920, height: 1080 },
		deviceScaleFactor: scale,
	});
	const page = await context.newPage();
	page.on("pageerror", (err) => console.error("page error:", err));
	await page.goto(url);
	const meta = await page.evaluate(() => window.__ready);
	const cdp = await context.newCDPSession(page);
	return { page, cdp, meta, context };
}

async function capture(target, t, file) {
	await target.page.evaluate((time) => window.__seek(time), t);
	const { data } = await target.cdp.send("Page.captureScreenshot", {
		format: "png",
		optimizeForSpeed: true,
		clip: { x: 0, y: 0, width: 1920, height: 1080, scale },
	});
	writeFileSync(file, Buffer.from(data, "base64"));
}

async function main() {
	const server = await serve(docsRoot);
	const { port } = server.address();
	const url = `http://127.0.0.1:${port}/video/composition.html?render`;
	const browser = await chromium.launch({
		executablePath: process.env.CHROMIUM_PATH || undefined,
		args: ["--force-color-profile=srgb", "--font-render-hinting=none", "--disable-lcd-text"],
	});

	try {
		if (opts.stills) {
			const dir = path.join(here, "stills");
			mkdirSync(dir, { recursive: true });
			const target = await openPage(browser, url);
			for (const s of opts.stills.split(",")) {
				const t = Number(s);
				const file = path.join(dir, `t${t.toFixed(2).padStart(6, "0")}.png`);
				await capture(target, t, file);
				console.log(file);
			}
			return;
		}

		const probe = await openPage(browser, url);
		const { duration, fps } = probe.meta;
		await probe.context.close();
		const total = Math.round(duration * fps);
		const framesDir = opts.frames;

		if (!opts["skip-capture"]) {
			rmSync(framesDir, { recursive: true, force: true });
			mkdirSync(framesDir, { recursive: true });
			const workers = Number(opts.workers);
			const per = Math.ceil(total / workers);
			let done = 0;
			const started = Date.now();
			await Promise.all(
				Array.from({ length: workers }, async (_, w) => {
					const target = await openPage(browser, url);
					for (let f = w * per; f < Math.min(total, (w + 1) * per); f++) {
						await capture(target, f / fps, path.join(framesDir, `f${String(f).padStart(5, "0")}.png`));
						done++;
						if (done % 120 === 0) {
							const rate = done / ((Date.now() - started) / 1000);
							console.log(`captured ${done}/${total} frames (${rate.toFixed(1)} fps)`);
						}
					}
					await target.context.close();
				}),
			);
		}

		mkdirSync(opts.out, { recursive: true });
		const input = ["-y", "-hide_banner", "-loglevel", "warning", "-framerate", String(fps), "-i", path.join(framesDir, "f%05d.png")];
		const color = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"];
		const vf = "scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format=yuv420p";
		const av1 = path.join(opts.out, `${opts.name}-av1.mp4`);
		const mp4 = path.join(opts.out, `${opts.name}.mp4`);
		const poster = path.join(opts.out, `${opts.name}-poster.jpg`);
		const gop = String(fps * 2);

		// AV1 is a third smaller for this flat, text-heavy content; H.264 covers
		// browsers without AV1 decoding (older Safari and iOS).
		console.log("encoding", av1);
		await run(FFMPEG, [...input, "-vf", vf, "-c:v", "libaom-av1", "-crf", "34", "-b:v", "0", "-cpu-used", "6", "-row-mt", "1", "-tiles", "2x2", "-aq-mode", "0", "-aom-params", "tune-content=screen", "-g", gop, ...color, "-movflags", "+faststart", "-an", av1]);
		console.log("encoding", mp4);
		await run(FFMPEG, [...input, "-vf", vf, "-c:v", "libx264", "-preset", "veryslow", "-crf", "21", "-tune", "animation", "-profile:v", "high", "-level:v", "4.2", "-g", gop, ...color, "-movflags", "+faststart", "-an", mp4]);

		const posterFrame = Math.min(total - 1, Math.round(Number(opts.poster) * fps));
		await run(FFMPEG, ["-y", "-hide_banner", "-loglevel", "warning", "-i", path.join(framesDir, `f${String(posterFrame).padStart(5, "0")}.png`), "-vf", "scale=1920:1080:flags=lanczos", "-frames:v", "1", "-update", "1", "-q:v", "3", poster]);

		for (const file of [av1, mp4, poster]) {
			console.log(`${path.relative(docsRoot, file)}  ${(statSync(file).size / 1e6).toFixed(2)} MB`);
		}
		if (!opts["keep-frames"]) rmSync(framesDir, { recursive: true, force: true });
	} finally {
		await browser.close();
		server.close();
	}
}

main().catch((err) => {
	console.error(err);
	process.exit(1);
});
