/*
 * SQLsaber feature video.
 *
 * Every frame is a pure function of time: window.__seek(t) lays out the whole
 * stage for second `t`, so render.mjs can step through the video frame by
 * frame and get identical output on every run.
 *
 * Open composition.html through a local server to preview in real time
 * (space: pause, arrow keys: seek, ?t=12.5 freezes a single frame).
 */
(() => {
	"use strict";

	const W = 1920;
	const H = 1080;
	const FPS = 60;

	// Scene windows on the global timeline, in seconds. Overlaps are transitions.
	const T = {
		intro: [0.0, 3.6],
		ask: [3.2, 7.65],
		demo: [6.8, 19.3],
		safe: [19.15, 23.7],
		stack: [23.55, 29.4],
		more: [29.25, 34.65],
		outro: [34.5, 40.3],
	};
	const DURATION = 40.3;
	// The question flies from the ASK scene into the terminal at this moment.
	const MORPH = T.demo[0];
	const FLIGHT = { stagger: 0.024, dur: 0.62 };

	// ------------------------------------------------------------------ math

	const clamp = (x, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, x));
	const lerp = (a, b, p) => a + (b - a) * p;
	const seg = (t, a, b) => clamp((t - a) / (b - a));
	const E = {
		outCubic: (p) => 1 - (1 - p) ** 3,
		inCubic: (p) => p ** 3,
		inOutCubic: (p) => (p < 0.5 ? 4 * p ** 3 : 1 - (-2 * p + 2) ** 3 / 2),
		outQuart: (p) => 1 - (1 - p) ** 4,
	};
	// Eased progress of question word `i` on its way into the terminal.
	const flight = (i, g) => E.inOutCubic(seg(g, MORPH + i * FLIGHT.stagger, MORPH + i * FLIGHT.stagger + FLIGHT.dur));
	// Fade in over [a, b] and out over [c, d].
	const inOut = (t, a, b, c, d) => Math.min(seg(t, a, b), 1 - seg(t, c, d));
	const hash = (n) => {
		let x = Math.imul((n | 0) ^ 0x9e3779b9, 0x85ebca6b);
		x ^= x >>> 13;
		x = Math.imul(x, 0xc2b2ae35);
		x ^= x >>> 16;
		return (x >>> 0) / 4294967296;
	};

	// ------------------------------------------------------------------- DOM

	function h(tag, props, ...kids) {
		const node = document.createElement(tag);
		if (props) {
			for (const [k, v] of Object.entries(props)) {
				if (k === "class") node.className = v;
				else if (k === "style") Object.assign(node.style, v);
				else if (k === "html") node.innerHTML = v;
				else if (k === "text") node.textContent = v;
				else node.setAttribute(k, v);
			}
		}
		for (const kid of kids.flat()) {
			if (kid !== null && kid !== undefined && kid !== false) node.append(kid);
		}
		return node;
	}
	const px = (n) => `${n}px`;
	function place(node, x, y) {
		node.style.left = px(x);
		node.style.top = px(y);
		return node;
	}
	function op(node, v) {
		const s = String(Math.round(clamp(v) * 1000) / 1000);
		if (node._op !== s) {
			node._op = s;
			node.style.opacity = s;
		}
	}
	function tf(node, v) {
		if (node._tf !== v) {
			node._tf = v;
			node.style.transform = v;
		}
	}
	function show(node, visible) {
		const v = visible ? "" : "hidden";
		if (node._vis !== v) {
			node._vis = v;
			node.style.visibility = v;
		}
	}
	function label(text) {
		const txt = h("span");
		const node = h("div", { class: "label abs" }, h("span", { class: "sig" }), txt);
		node._text = text;
		node._txt = txt;
		return node;
	}

	// ----------------------------------------------------------------- icons

	const SVG_NS = "http://www.w3.org/2000/svg";
	const ICON = {
		check: (size = 22, color = "#fff") =>
			`<svg width="${size}" height="${size}" viewBox="0 0 22 22"><path d="M3.5 11.5l5 5 10-11" fill="none" stroke="${color}" stroke-width="2.4" stroke-linecap="square"/></svg>`,
		cross: (size = 22, color = "#ffffff") =>
			`<svg width="${size}" height="${size}" viewBox="0 0 22 22"><path d="M5 5l12 12M17 5L5 17" fill="none" stroke="${color}" stroke-width="2.8" stroke-linecap="square"/></svg>`,
		spinner: () => {
			let s = '<svg width="22" height="22" viewBox="0 0 22 22">';
			for (let i = 0; i < 8; i++) {
				const a = (i / 8) * Math.PI * 2 - Math.PI / 2;
				s += `<circle cx="${(11 + Math.cos(a) * 7.6).toFixed(2)}" cy="${(11 + Math.sin(a) * 7.6).toFixed(2)}" r="2.1" fill="#fff"/>`;
			}
			return `${s}</svg>`;
		},
	};
	function spin(ico, t) {
		const phase = Math.floor(t * 14) % 8;
		ico._dots ??= [...ico.querySelectorAll("circle")];
		ico._dots.forEach((d, i) => {
			d.setAttribute("opacity", String(Math.max(0.16, 1 - ((phase - i + 8) % 8) * 0.21)));
		});
	}

	// ------------------------------------------------------------ text tools

	// Typed text that keeps its final layout: untyped characters stay in place
	// as transparent "ghost" text, so nothing reflows while typing.
	class Typed {
		constructor(parent, segs) {
			this.parts = segs.map(([text, cls]) => {
				const vis = h("span", { class: cls || "" });
				const ghost = h("span", { class: `${cls || ""} ghost`, text });
				parent.append(vis, ghost);
				return { text, vis, ghost };
			});
			this.length = segs.reduce((n, [text]) => n + text.length, 0);
			this.shown = -1;
		}
		show(n) {
			n = Math.max(0, Math.min(this.length, Math.floor(n)));
			if (n === this.shown) return;
			this.shown = n;
			let rest = n;
			for (const p of this.parts) {
				const k = Math.max(0, Math.min(p.text.length, rest));
				p.vis.textContent = p.text.slice(0, k);
				p.ghost.textContent = p.text.slice(k);
				rest -= p.text.length;
			}
		}
	}

	// Chunked, slightly irregular reveal that reads like streamed model output.
	function streamPlan(total, t0, dur, seed) {
		const cuts = [0];
		let k = 0;
		while (cuts[cuts.length - 1] < total) {
			cuts.push(Math.min(total, cuts[cuts.length - 1] + 2 + Math.floor(hash(seed + k++) * 6)));
		}
		const times = cuts.map((c, i) => t0 + (c / total) * dur + (i ? (hash(seed * 3 + i) - 0.5) * 0.03 : 0));
		return (t) => {
			let n = 0;
			for (let i = 0; i < cuts.length; i++) if (t >= times[i]) n = cuts[i];
			return n;
		};
	}

	const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&*+=/<>";
	// Characters flicker through random glyphs before settling on the text.
	function scramble(node, text, t, t0, seed = 1, stagger = 0.024, settle = 0.18) {
		let out = "";
		for (let i = 0; i < text.length; i++) {
			const ti = t0 + i * stagger;
			const ch = text[i];
			if (t < ti) out += " ";
			else if (ch === " " || t >= ti + settle) out += ch;
			else out += GLYPHS[Math.floor(hash(seed * 997 + i * 31 + Math.floor(t * 30)) * GLYPHS.length)];
		}
		if (node._s !== out) {
			node._s = out;
			node.textContent = out;
		}
	}
	function renderLabel(node, t, t0, seed) {
		scramble(node._txt, node._text, t, t0, seed);
		op(node.firstChild, seg(t, t0 - 0.05, t0 + 0.05));
	}

	// Headline words that slide up from behind a mask.
	function riseWords(node, text) {
		const inners = [];
		const words = text.split(" ");
		words.forEach((word, i) => {
			const inner = h("span", { text: word });
			node.append(h("span", { class: "mask" }, inner));
			if (i < words.length - 1) node.append(" ");
			inners.push(inner);
		});
		return inners;
	}
	function renderRise(inners, t, t0, stagger = 0.06, dur = 0.75) {
		inners.forEach((inner, i) => {
			const p = E.outQuart(seg(t, t0 + i * stagger, t0 + i * stagger + dur));
			tf(inner, `translateY(${((1 - p) * 110).toFixed(2)}%)`);
		});
	}

	const SQL_KW = new Set(
		"SELECT FROM JOIN ON WHERE AND OR AS IN BETWEEN OVER PARTITION BY ORDER GROUP HAVING LIMIT WITH NOT NULL IS".split(" "),
	);
	const SQL_FN = new Set(["ROW_NUMBER", "COUNT", "MIN", "MAX", "SUM", "AVG"]);
	function sqlTokens(line) {
		const out = [];
		const re = /('(?:[^']|'')*')|(\b\d+\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|(.)/g;
		let m = re.exec(line);
		while (m) {
			const [tok, str, num, word, ws] = m;
			let cls = "t-op";
			if (str) cls = "t-str";
			else if (num) cls = "t-num";
			else if (word) {
				const up = word.toUpperCase();
				cls = SQL_KW.has(up) ? "t-kw" : SQL_FN.has(up) ? "t-fn" : "t-id";
			} else if (ws) cls = "";
			out.push([tok, cls]);
			m = re.exec(line);
		}
		return out;
	}

	// Greedy word wrap for monospace segments: [[text, cls], ...] -> lines.
	function wrapSegments(segs, width) {
		const words = [];
		for (const [text, cls] of segs) {
			for (const piece of text.split(/(\s+)/)) if (piece) words.push([piece, cls]);
		}
		const lines = [[]];
		let col = 0;
		for (const [piece, cls] of words) {
			const space = /^\s+$/.test(piece);
			if (!space && col + piece.length > width && col > 0) {
				const last = lines[lines.length - 1];
				while (last.length && /^\s+$/.test(last[last.length - 1][0])) last.pop();
				lines.push([]);
				col = 0;
			}
			if (space && col === 0) continue;
			lines[lines.length - 1].push([piece, cls]);
			col += piece.length;
		}
		return lines;
	}

	// ------------------------------------------------------------ dot matrix

	// Rasterise the [SQL]saber lockup (the favicon mark) onto a grid of cells.
	// Inside the box, "SQL" is cut out of lit dots; "saber" is drawn in lit dots.
	function buildLockupGrid() {
		const k = 12;
		const rows = 21;
		const cap = 12;
		const size = (cap / 0.7) * k;
		const cv = document.createElement("canvas");
		const cx = cv.getContext("2d");
		// Letters are placed one by one with extra tracking so that the cut-out
		// "SQL" stays legible at dot resolution.
		const layout = (text, weight, track) => {
			cx.font = `${weight} ${size}px "Space Grotesk"`;
			let x = 0;
			const glyphs = [...text].map((ch) => {
				const g = { ch, x };
				x += cx.measureText(ch).width / k + track;
				return g;
			});
			return { glyphs, width: x - track, weight };
		};
		const sql = layout("SQL", 600, 1.7);
		const saber = layout("saber", 500, 0.9);
		const pad = 3.5;
		const gap = 2.5;
		const boxCols = Math.round(sql.width + 2 * pad);
		const saberX = boxCols + gap;
		const cols = Math.ceil(saberX + saber.width) + 1;
		cv.width = cols * k;
		cv.height = rows * k;
		const base = ((rows + cap) / 2) * k;
		cx.fillStyle = "#fff";
		const draw = (word, x0) => {
			cx.font = `${word.weight} ${size}px "Space Grotesk"`;
			for (const g of word.glyphs) cx.fillText(g.ch, (x0 + g.x) * k, base);
		};
		draw(sql, (boxCols - sql.width) / 2);
		draw(saber, saberX);
		const data = cx.getImageData(0, 0, cv.width, cv.height).data;
		const lit = new Uint8Array(cols * rows);
		for (let r = 0; r < rows; r++) {
			for (let c = 0; c < cols; c++) {
				let sum = 0;
				for (let y = 0; y < k; y++) {
					const off = ((r * k + y) * cv.width + c * k) * 4 + 3;
					for (let x = 0; x < k; x++) sum += data[off + x * 4];
				}
				const cov = sum / (k * k * 255);
				lit[r * cols + c] = c < boxCols ? (cov < 0.5 ? 1 : 0) : cov > 0.42 ? 1 : 0;
			}
		}
		return { cols, rows, lit };
	}

	// Draw the lockup as an LED panel: dim dots everywhere, lit dots switching
	// on in a left-to-right wave with a brief bloom.
	function drawLockup(ctx, g, o) {
		const { pitch, t } = o;
		const m = 2;
		const rDim = pitch * 0.26;
		const rLit = pitch * 0.37;
		const alpha = o.alpha ?? 1;
		const glows = [];
		for (let r = -m; r < g.rows + m; r++) {
			for (let c = -m; c < g.cols + m; c++) {
				const x = o.x + (c + 0.5) * pitch;
				const y = o.y + (r + 0.5) * pitch;
				const n = (r + 64) * 4096 + (c + 64);
				const dist = Math.hypot((c - g.cols / 2) / g.cols, ((r - g.rows / 2) / g.rows) * 0.35);
				const dim = seg(t, o.panel + dist * 0.5 + hash(n) * 0.15, o.panel + dist * 0.5 + hash(n) * 0.15 + 0.25);
				const lit = r >= 0 && r < g.rows && c >= 0 && c < g.cols && g.lit[r * g.cols + c];
				let on = 0;
				let fresh = 0;
				if (lit) {
					const tOn = o.on + o.span * (c / g.cols) + 0.1 * hash(n + 11) + 0.05 * (r / g.rows);
					on = seg(t, tOn, tOn + 0.34);
					fresh = on > 0 && on < 1 ? 1 - E.outCubic(on) : 0;
					if (o.off !== undefined) {
						const tOff = o.off + o.offSpan * (1 - c / g.cols) + 0.08 * hash(n + 5);
						on *= 1 - seg(t, tOff, tOff + 0.22);
					}
				}
				const dimA = dim * (1 - on) * alpha;
				if (dimA > 0.004) {
					ctx.globalAlpha = dimA;
					ctx.fillStyle = "#1d1d1d";
					ctx.beginPath();
					ctx.arc(x, y, rDim, 0, Math.PI * 2);
					ctx.fill();
				}
				if (on > 0.004) {
					const k = E.outCubic(on);
					ctx.globalAlpha = Math.min(1, on * 3) * alpha;
					const v = Math.round(lerp(255, 244, k));
					ctx.fillStyle = `rgb(${v},${v},${v})`;
					ctx.beginPath();
					ctx.arc(x, y, rLit * (1 + 0.45 * fresh), 0, Math.PI * 2);
					ctx.fill();
					if (fresh > 0.02) glows.push([x, y, fresh]);
				}
			}
		}
		ctx.globalCompositeOperation = "lighter";
		for (const [x, y, f] of glows) {
			ctx.globalAlpha = 0.14 * f * alpha;
			ctx.fillStyle = "#ffffff";
			ctx.beginPath();
			ctx.arc(x, y, rLit * 2.4, 0, Math.PI * 2);
			ctx.fill();
		}
		ctx.globalCompositeOperation = "source-over";
		ctx.globalAlpha = 1;
	}

	// ---------------------------------------------------------------- scenes

	let DPR = 1;
	const stage = document.getElementById("stage");
	const bgGrid = document.getElementById("bg-grid");

	function fxCanvas(parent) {
		const cv = h("canvas", { class: "fx" });
		cv.width = W * DPR;
		cv.height = H * DPR;
		parent.append(cv);
		const ctx = cv.getContext("2d");
		return ctx;
	}

	function sceneIntro(grid) {
		const el = h("div", { class: "scene" });
		const ctx = fxCanvas(el);
		const pitch = 13;
		const lw = grid.cols * pitch;
		const lh = grid.rows * pitch;
		const lx = Math.round((W - lw) / 2);
		const ly = Math.round(H / 2 - lh / 2 - 30);
		const tag = h("div", { class: "label abs", style: { left: "0", width: px(W), textAlign: "center", paddingLeft: "0.2em", fontSize: "24px", letterSpacing: "0.34em", color: "#9a9a9a" } });
		place(tag, 0, ly + lh + 64);
		el.append(tag);

		return {
			el,
			range: T.intro,
			render(t) {
				ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
				ctx.clearRect(0, 0, W, H);
				// Exit: the dots switch off in a quick wave, then the panel fades.
				const fade = E.inCubic(seg(t, 3.0, 3.4));
				drawLockup(ctx, grid, { x: lx, y: ly, pitch, t, panel: 0.0, on: 0.28, span: 0.8, off: 2.75, offSpan: 0.4, alpha: 1 - fade });
				scramble(tag, "AGENTIC SQL ASSISTANT", t, 1.15, 3, 0.03, 0.2);
				op(tag, 1 - E.inCubic(seg(t, 2.8, 3.2)));
			},
		};
	}

	function sceneAsk() {
		const el = h("div", { class: "scene" });
		const lab = label("Ask in plain English");
		place(lab, 160, 322);
		const q = h("div", { class: "question abs" });
		place(q, 154, 384);
		const lines = [
			["How", "many", "VPs", "became", "president"],
			["by", "election", "in", "the", "20th", "century?"],
		];
		const words = [];
		const chars = [];
		lines.forEach((ws, li) => {
			ws.forEach((w, wi) => {
				const wEl = h("span", { class: "w" });
				for (const ch of w) {
					const c = h("span", { class: "c", text: ch });
					wEl.append(c);
					chars.push(c);
				}
				q.append(wEl);
				words.push(wEl);
				if (wi < ws.length - 1) {
					q.append(" ");
					chars.push(null);
				}
			});
			if (li < lines.length - 1) {
				q.append(h("br"));
				chars.push(null);
			}
		});
		const cursor = h("div", { class: "cursor-block" });
		el.append(lab, q, cursor);

		const TYPE0 = 0.6;
		const CPS = 29;
		const charTime = chars.map((_, i) => TYPE0 + i / CPS + (hash(i * 7 + 1) - 0.5) * 0.03);
		const typeEnd = charTime[charTime.length - 1];
		let charRects = [];
		let flights = [];

		return {
			el,
			range: T.ask,
			measure(targets) {
				const base = stage.getBoundingClientRect();
				const rel = (r) => ({ x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height });
				charRects = chars.map((c) => (c ? rel(c.getBoundingClientRect()) : null));
				flights = words.map((w, i) => {
					const a = rel(w.getBoundingClientRect());
					const b = targets[i];
					return { dx: b.x - a.x, dy: b.y - a.y, sx: b.w / a.w, sy: b.h / a.h };
				});
			},
			render(t, g) {
				op(el, seg(t, 0, 0.35));
				renderLabel(lab, t, 0.3, 5);
				op(lab, 1 - seg(g, MORPH - 0.05, MORPH + 0.25));

				chars.forEach((c, i) => {
					if (!c) return;
					const p = seg(t, charTime[i], charTime[i] + 0.09);
					op(c, p);
					c.style.top = px(Math.round((1 - E.outCubic(p)) * 12));
				});

				// Cursor: blinks before and after typing, solid while typing.
				let last = -1;
				for (let i = 0; i < chars.length; i++) if (chars[i] && t >= charTime[i]) last = i;
				const typing = t >= TYPE0 && t <= typeEnd + 0.1;
				const blink = typing || Math.floor((t - TYPE0) * 2.2) % 2 === 0;
				const ref = charRects[last >= 0 ? last : 0];
				const cx = last >= 0 ? ref.x + ref.w + 8 : ref.x - 4;
				place(cursor, Math.round(cx), Math.round(ref.y + ref.h * 0.19));
				cursor.style.width = "12px";
				cursor.style.height = px(Math.round(ref.h * 0.64));
				op(cursor, t >= 0.3 && g < MORPH && blink ? 1 : 0);

				// Morph: every word flies into its slot on the terminal command line.
				words.forEach((w, i) => {
					const f = flights[i];
					if (!f) return;
					const p = flight(i, g);
					if (p <= 0) {
						tf(w, "none");
						op(w, 1);
						return;
					}
					const arc = -Math.sin(Math.PI * p) * 46;
					tf(w, `translate(${(f.dx * p).toFixed(2)}px, ${(f.dy * p + arc).toFixed(2)}px) scale(${lerp(1, f.sx, p).toFixed(4)}, ${lerp(1, f.sy, p).toFixed(4)})`);
					op(w, 1 - seg(p, 0.72, 1));
				});
			},
		};
	}

	function sceneDemo() {
		const el = h("div", { class: "scene" });
		const TX = 96;
		const TY = 96;
		const TW = 1180;
		const TH = 888;
		const VIEW_H = TH - 53 - 26 - 30;
		const term = h("div", { class: "term", style: { left: px(TX), top: px(TY), width: px(TW), height: px(TH) } });
		const bg = h("div", { class: "term-bg" });
		const frame = document.createElementNS(SVG_NS, "svg");
		frame.setAttribute("class", "term-frame");
		frame.setAttribute("width", String(TW));
		frame.setAttribute("height", String(TH));
		frame.innerHTML = `<rect x="0.5" y="0.5" width="${TW - 1}" height="${TH - 1}" fill="none" stroke="#2e2e2e" stroke-width="1" pathLength="1000" stroke-dasharray="1000 1000" stroke-dashoffset="1000"/>`;
		const frameRect = frame.firstChild;
		const bar = h("div", { class: "term-bar" }, h("span", { class: "lights" }, h("i"), h("i"), h("i")), h("span", { class: "title", text: "saber — ~/data" }));
		const view = h("div", { class: "term-view" });
		const content = h("div", { class: "term-content" });
		const tcursor = h("div", { class: "term-cursor" });
		content.append(tcursor);
		view.append(content);
		term.append(bg, frame, bar, view);
		el.append(term);

		const rows = [];
		const row = (height = 40) => {
			const r = h("div", { class: "row" });
			if (height !== 40) r.style.height = px(height);
			content.insertBefore(r, tcursor);
			rows.push(r);
			return r;
		};

		// Rows 0-1: the command. The question words land here during the morph.
		const Q = ["How", "many", "VPs", "became", "president", "by", "election", "in", "the", "20th", "century?"];
		const r0 = row();
		const prefix = new Typed(r0, [
			["$ ", "t-p"],
			['saber -d ./legislators.db "', "t-cmd"],
		]);
		const qWords = Q.map((w) => h("span", { class: "t-q", text: w }));
		qWords.slice(0, 6).forEach((w, i) => {
			r0.append(w);
			if (i < 5) r0.append(" ");
		});
		const r1 = row();
		qWords.slice(6).forEach((w, i) => {
			r1.append(w);
			if (i < 4) r1.append(" ");
		});
		const closeQuote = h("span", { class: "t-cmd", text: '"' });
		r1.append(closeQuote);
		op(closeQuote, 0);

		const r2 = row();
		const connected = new Typed(r2, [
			["• ", "t-dim"],
			["Connected to", "t-b"],
			[": legislators (SQLite)", ""],
		]);
		row();

		function toolRow(text, detail) {
			const r = row();
			const ico = h("span", { class: "ico", html: ICON.spinner() });
			const done = h("span", { class: "ico", html: ICON.check(22, "#fff") });
			done.style.position = "absolute";
			done.style.left = "0";
			r.append(ico, done);
			const title = new Typed(r, [[text, "t-b"]]);
			const d = row();
			const det = new Typed(d, detail);
			return { r, ico, done, title, det };
		}
		const tool1 = toolRow("Discovering available tables", [
			["  Database Tables (6 total)", "t-dimb"],
			[" · executives, executive_terms, …", "t-dim"],
		]);
		const tool2 = toolRow("Examining schema", [
			["  Schema Information (2 tables)", "t-dimb"],
			[" · executives, executive_terms", "t-dim"],
		]);
		row();
		const execRow = row();
		const execTitle = new Typed(execRow, [["Executing SQL:", "t-b"]]);

		const SQL = [
			"SELECT e.name, t.start AS took_office",
			"FROM (",
			"  SELECT *, ROW_NUMBER() OVER (",
			"    PARTITION BY executive_id ORDER BY start) AS n",
			"  FROM executive_terms WHERE type = 'prez'",
			") t",
			"JOIN executives e ON e.id = t.executive_id",
			"WHERE t.n = 1 AND t.how = 'election'",
			"  AND t.start BETWEEN '1901-01-01' AND '2000-12-31'",
			"  AND e.id IN (SELECT executive_id FROM executive_terms",
			"               WHERE type = 'viceprez');",
		];
		const sqlRows = SQL.map((line) => {
			const r = row();
			return { r, typed: new Typed(r, sqlTokens(line)), len: line.length };
		});
		row();
		const resRow = row();
		const resTitle = new Typed(resRow, [["Results (2 rows):", "t-b"]]);
		const tblRow = row(132);
		const tbl = h("div", { class: "tbl" });
		const tLines = [
			["name", "took_office"],
			["Richard Nixon", "1969-01-20"],
			["George Bush", "1989-01-20"],
		].map((cells, i) => {
			const tr = h("div", { class: i === 0 ? "tr th" : "tr" }, ...cells.map((c) => h("span", { text: c })));
			tbl.append(tr);
			return [...tr.children];
		});
		tblRow.append(tbl);
		row();

		// Two paragraphs: the answer, then how it was derived (matches the SQL above).
		const ANSWER = [
			[
				["2 vice presidents", "t-b"],
				[" won the presidency by election in the 20th century: ", ""],
				["Richard Nixon", "t-b"],
				[" (1969) and ", ""],
				["George H. W. Bush", "t-b"],
				[" (1989).", ""],
			],
			[["I counted each president's first term, kept those won by election in 1901–2000, and required earlier VP service.", ""]],
		];
		const ansLines = [];
		ANSWER.forEach((para, i) => {
			if (i > 0) row();
			for (const segs of wrapSegments(para, 62)) {
				const r = row();
				ansLines.push({ r, typed: new Typed(r, segs) });
			}
		});
		const status = h("div", { class: "abs", style: { left: "0", top: "0" } });
		const statusIco = h("span", { class: "ico", html: ICON.spinner() });
		status.append(statusIco, h("span", { class: "t-dim", text: "Crunching data..." }));
		ansLines[0].r.append(status);

		// Right-hand panel: what the agent is doing, in plain words.
		const stepsLab = label("How it works");
		stepsLab.classList.add("steps-label");
		el.append(stepsLab);
		const STEPS = [
			["01", "Reads your schema", "list_tables · introspect_schema"],
			["02", "Writes the SQL", "Tailored to SQLite"],
			["03", "Runs it read-only", "SELECT-only by default"],
			["04", "Explains the result", "In plain English"],
		];
		const STEP_Y = 170;
		const STEP_DY = 116;
		const rail = h("div", { class: "rail", style: { top: px(STEP_Y + 30), height: px(STEP_DY * 3 - 30) } });
		const railFill = h("i");
		rail.append(railFill);
		el.append(rail);
		const steps = STEPS.map(([ix, title, sub], i) => {
			const mk = h("div", { class: "mk" }, h("i"), h("b"), h("u"));
			const node = h("div", { class: "step" }, mk, h("div", { class: "ix", text: ix }), h("div", { class: "tt", text: title }), h("div", { class: "sb", text: sub }));
			node.style.top = px(STEP_Y + i * STEP_DY);
			el.append(node);
			return { node, box: mk.children[0], fill: mk.children[1], check: mk.children[2] };
		});

		const card = h("div", { class: "answer-card abs" });
		const cardLab = label("Answer");
		cardLab.classList.remove("abs");
		const big = h("div", { class: "big", text: "2" });
		const cap = h("div", { class: "cap", html: "VPs who became president<br>by election, 1901–2000" });
		card.append(cardLab, big, cap);
		el.append(card);

		// Timeline (seconds from the start of the scene).
		const S = {
			frame: [0.0, 0.55],
			prefix: [0.1, 0.36],
			quote: 0.84,
			connected: 1.0,
			tool1: [1.3, 1.95],
			tool2: [2.2, 2.8],
			exec: 3.08,
			sql: [3.2, 5.4],
			results: 5.9,
			crunch: [6.45, 7.1],
			answer: [7.1, 9.3],
			card: 9.45,
			exit: [11.9, 12.5],
		};
		const stepTimes = [S.tool1[0], S.exec, 5.5, S.answer[0], S.answer[1] + 0.1];
		const sqlTotal = sqlRows.reduce((n, s) => n + s.len, 0);
		const sqlStream = streamPlan(sqlTotal, S.sql[0], S.sql[1] - S.sql[0], 101);
		const ansTotal = ansLines.reduce((n, a) => n + a.typed.length, 0);
		const ansStream = streamPlan(ansTotal, S.answer[0], S.answer[1] - S.answer[0], 202);

		// When each row first shows something; drives the terminal scroll.
		const reveal = new Map();
		let scrollSteps = [];

		function rowTimeOfChar(total, idx, t0, dur) {
			return t0 + (idx / total) * dur;
		}

		return {
			el,
			range: T.demo,
			measure() {
				const base = stage.getBoundingClientRect();
				const targets = qWords.map((w) => {
					const r = w.getBoundingClientRect();
					return { x: r.left - base.left, y: r.top - base.top, w: r.width, h: r.height };
				});
				reveal.set(rows.indexOf(tool1.r), S.tool1[0]);
				reveal.set(rows.indexOf(tool1.det.parts[0].vis.parentNode), S.tool1[1]);
				reveal.set(rows.indexOf(tool2.r), S.tool2[0]);
				reveal.set(rows.indexOf(tool2.det.parts[0].vis.parentNode), S.tool2[1]);
				reveal.set(rows.indexOf(execRow), S.exec);
				let acc = 0;
				for (const s of sqlRows) {
					reveal.set(rows.indexOf(s.r), rowTimeOfChar(sqlTotal, acc, S.sql[0], S.sql[1] - S.sql[0]));
					acc += s.len;
				}
				reveal.set(rows.indexOf(resRow), S.results);
				reveal.set(rows.indexOf(tblRow), S.results + 0.1);
				acc = 0;
				ansLines.forEach((a, i) => {
					const tRow = i === 0 ? S.crunch[0] : rowTimeOfChar(ansTotal, acc, S.answer[0], S.answer[1] - S.answer[0]);
					reveal.set(rows.indexOf(a.r), tRow);
					acc += a.typed.length;
				});
				// Superimposed eased steps keep the scroll smooth when rows arrive quickly.
				let prev = 0;
				scrollSteps = [...reveal.entries()]
					.sort((a, b) => a[1] - b[1])
					.map(([i, time]) => {
						const r = rows[i];
						const need = Math.max(0, r.offsetTop + r.offsetHeight - VIEW_H);
						const step = Math.max(0, need - prev);
						prev = Math.max(prev, need);
						return [time, step];
					})
					.filter(([, step]) => step > 0);
				return targets;
			},
			render(t) {
				// Exit: quick fade with a slight lift.
				const exit = E.inCubic(seg(t, S.exit[0], S.exit[1]));
				op(el, 1 - exit);
				tf(el, exit > 0 ? `translateY(${(-exit * 24).toFixed(2)}px)` : "none");

				const fp = E.inOutCubic(seg(t, S.frame[0], S.frame[1]));
				frameRect.setAttribute("stroke-dashoffset", String(1000 * (1 - fp)));
				op(bg, E.outCubic(seg(t, 0.05, 0.45)));
				op(bar, seg(t, 0.25, 0.55));

				let scroll = 0;
				for (const [time, step] of scrollSteps) scroll += step * E.inOutCubic(seg(t, time - 0.12, time + 0.26));
				tf(content, `translateY(${(-scroll).toFixed(2)}px)`);

				prefix.show(seg(t, S.prefix[0], S.prefix[1]) * prefix.length);
				qWords.forEach((w, i) => op(w, seg(flight(i, t + MORPH), 0.72, 1)));
				op(closeQuote, seg(t, S.quote, S.quote + 0.05));
				connected.show(seg(t, S.connected, S.connected + 0.22) * connected.length);

				for (const [tool, [a, b]] of [
					[tool1, S.tool1],
					[tool2, S.tool2],
				]) {
					const started = t >= a;
					const finished = t >= b;
					show(tool.ico, started && !finished);
					show(tool.done, finished);
					if (started && !finished) spin(tool.ico, t);
					tool.title.show(seg(t, a, a + 0.16) * tool.title.length);
					tool.det.show(seg(t, b, b + 0.2) * tool.det.length);
				}
				execTitle.show(seg(t, S.exec, S.exec + 0.12) * execTitle.length);

				let n = sqlStream(t);
				let cursorRow = -1;
				let cursorCol = 0;
				for (const s of sqlRows) {
					const k = Math.min(s.len, Math.max(0, n));
					s.typed.show(k);
					if (k > 0 && t < S.sql[1] + 0.2) {
						cursorRow = rows.indexOf(s.r);
						cursorCol = k;
					}
					n -= s.len;
				}

				resTitle.show(seg(t, S.results, S.results + 0.12) * resTitle.length);
				tLines.forEach((cells, i) => {
					const p = E.outCubic(seg(t, S.results + 0.1 + i * 0.09, S.results + 0.3 + i * 0.09));
					for (const c of cells) op(c, p);
				});
				op(tbl, t >= S.results + 0.1 ? 1 : 0);

				show(status, t >= S.crunch[0] && t < S.crunch[1]);
				if (t >= S.crunch[0] && t < S.crunch[1]) spin(statusIco, t);
				let an = ansStream(t);
				for (const a of ansLines) {
					const k = Math.min(a.typed.length, Math.max(0, an));
					a.typed.show(k);
					if (k > 0 && t < S.answer[1] + 0.25) {
						cursorRow = rows.indexOf(a.r);
						cursorCol = k;
					}
					an -= a.typed.length;
				}

				if (cursorRow >= 0) {
					const r = rows[cursorRow];
					place(tcursor, cursorCol * 17.136, r.offsetTop + 3);
				}
				op(tcursor, cursorRow >= 0 ? 1 : 0);

				// Steps panel.
				const panelIn = (i) => E.outCubic(seg(t, 0.95 + i * 0.07, 1.4 + i * 0.07));
				op(stepsLab, panelIn(0));
				renderLabel(stepsLab, t, 0.95, 9);
				op(rail, panelIn(1));
				let fill = 0;
				steps.forEach((st, i) => {
					const pin = panelIn(i + 1);
					const a = stepTimes[i];
					const b = stepTimes[i + 1];
					const active = seg(t, a, a + 0.2) * (1 - seg(t, b, b + 0.2));
					const done = seg(t, b, b + 0.2);
					op(st.node, pin * (0.3 + 0.7 * Math.max(active, done * 0.72)));
					tf(st.node, `translateX(${((1 - pin) * 24).toFixed(2)}px)`);
					op(st.fill, active * (0.75 + 0.25 * Math.cos((t - a) * 7)));
					op(st.box, 1 - done);
					op(st.check, done);
					if (t >= a) fill = i + seg(t, a, b);
				});
				railFill.style.height = px(Math.min(3, fill) * STEP_DY);

				// Answer card with a power-on flicker for the big number.
				const cp = seg(t, S.card, S.card + 0.5);
				op(card, cp > 0 ? 1 : 0);
				renderLabel(cardLab, t, S.card, 13);
				const ft = t - (S.card + 0.12);
				const flick = ft < 0 ? 0 : ft < 0.05 ? 1 : ft < 0.1 ? 0.15 : ft < 0.13 ? 1 : ft < 0.19 ? 0.3 : 1;
				op(big, flick);
				op(cap, E.outCubic(seg(t, S.card + 0.35, S.card + 0.8)));
				tf(cap, `translateY(${((1 - E.outCubic(seg(t, S.card + 0.35, S.card + 0.8))) * 14).toFixed(2)}px)`);
			},
		};
	}

	function sceneSafe() {
		const el = h("div", { class: "scene" });
		const lab = label("Safe by default");
		place(lab, 160, 238);
		const TEXT = "DROP TABLE legislators;";
		const X0 = 156;
		const Y0 = 290;
		const drop = h("div", { class: "drop abs" });
		place(drop, X0, Y0);
		const typed = new Typed(drop, [[TEXT, ""]]);
		const strike = h("div", { class: "strike abs" });
		const cursor = h("div", { class: "cursor-block" });
		const err = h("div", { class: "errline abs" });
		place(err, 160, 500);
		err.append(h("span", { class: "ico", html: ICON.cross(26) }));
		const errTyped = new Typed(err, [
			[" SQL error: ", "t-dim"],
			["Only SELECT-like statements are allowed", ""],
		]);
		const head = h("div", { class: "headline abs" });
		place(head, 154, 700);
		const headWords = riseWords(head, "Read-only by default.");
		const sub = h("div", { class: "sub abs", html: "Writes stay off unless you pass <code>--allow-dangerous</code>." });
		place(sub, 160, 830);
		el.append(drop, strike, lab, cursor, err, head, sub);

		let geo = null;
		const S = { type: [0.25, 0.95], strike: [1.3, 1.65], err: [1.85, 2.45], head: 2.35, sub: 2.75, exit: [4.1, 4.55] };

		return {
			el,
			range: T.safe,
			measure() {
				// A zero-size inline-block sits on the baseline of the statement.
				const probe = h("span", { style: { display: "inline-block", width: "0", height: "0" } });
				drop.append(probe);
				const baseline = probe.offsetTop;
				probe.remove();
				const w = drop.offsetWidth;
				const size = Number.parseFloat(getComputedStyle(drop).fontSize);
				geo = { w, hh: drop.offsetHeight, baseline, charW: w / TEXT.length };
				place(strike, X0 - 10, Math.round(Y0 + baseline - size * 0.31 - 3));
				strike.style.width = px(w + 20);
			},
			render(t) {
				const exit = E.inCubic(seg(t, S.exit[0], S.exit[1]));
				op(el, seg(t, 0, 0.25) * (1 - exit));
				renderLabel(lab, t, 0.1, 21);
				const n = seg(t, S.type[0], S.type[1]) * TEXT.length;
				typed.show(n);

				// The guard rejects the statement: strike it through and dim it.
				const sp = E.inOutCubic(seg(t, S.strike[0], S.strike[1]));
				tf(strike, `scaleX(${sp.toFixed(4)})`);
				op(strike, sp > 0 ? 1 : 0);
				const dim = E.outCubic(seg(t, S.strike[0] + 0.15, S.strike[1] + 0.5));
				const textGrey = Math.round(lerp(255, 78, dim));
				drop.style.color = `rgb(${textGrey},${textGrey},${textGrey})`;
				const strikeGrey = Math.round(lerp(240, 150, dim));
				strike.style.background = `rgb(${strikeGrey},${strikeGrey},${strikeGrey})`;

				// Typing cursor.
				const typing = t < S.strike[0];
				const blink = t < S.type[1] || Math.floor(t * 2.4) % 2 === 0;
				place(cursor, Math.round(X0 + Math.floor(n) * geo.charW + 6), Y0 + 30);
				cursor.style.width = "14px";
				cursor.style.height = px(geo.hh - 60);
				op(cursor, t > S.type[0] - 0.1 && typing && blink ? 1 : 0);

				errTyped.show(seg(t, S.err[0], S.err[1]) * errTyped.length);
				op(err.firstChild, seg(t, S.err[0], S.err[0] + 0.05));
				renderRise(headWords, t, S.head);
				op(sub, E.outCubic(seg(t, S.sub, S.sub + 0.5)));
				tf(sub, `translateY(${((1 - E.outCubic(seg(t, S.sub, S.sub + 0.5))) * 16).toFixed(2)}px)`);
			},
		};
	}

	function sceneStack() {
		const el = h("div", { class: "scene" });
		const labA = label("Connect your data");
		const labB = label("Any provider");
		place(labA, 160, 128);
		place(labB, 160, 128);
		const headA = h("div", { class: "headline abs" });
		const headB = h("div", { class: "headline abs" });
		place(headA, 154, 170);
		place(headB, 154, 170);
		const headAW = riseWords(headA, "Works with your stack.");
		const headBW = riseWords(headB, "Bring your own model.");
		el.append(labA, labB, headA, headB);

		const DBS = [
			["PostgreSQL", "postgresql://…"],
			["MySQL", "mysql://…"],
			["SQLite", "./app.db"],
			["DuckDB", "./warehouse.duckdb"],
			["CSV", "./customers.csv"],
			["Parquet", "./orders.parquet"],
		];
		const grid = h("div", { class: "abs", style: { left: "0", top: "0" } });
		el.append(grid);
		const tiles = DBS.map(([name, capText], i) => {
			const node = h("div", { class: "tile" }, h("div", { class: "nm", text: name }), h("div", { class: "cp", text: capText }), h("div", { class: "led" }), h("div", { class: "glow" }));
			place(node, 160 + (i % 3) * 544, 334 + Math.floor(i / 3) * 224);
			grid.append(node);
			return { node, led: node.querySelector(".led"), glow: node.querySelector(".glow") };
		});
		const strip = h("div", { class: "strip" });
		Object.assign(strip.style, { left: "160px", top: "802px", width: "1600px", height: "96px", padding: "0 36px", lineHeight: "94px" });
		const CMD = [
			["$ ", "t-p"],
			["saber -d ", ""],
			["./customers.csv", ""],
			[" -d ", ""],
			["./orders.parquet", ""],
			[' "Revenue by customer"', ""],
		];
		const cmd = new Typed(strip, CMD);
		const litAt = [2 + 9 + 15, 2 + 9 + 15 + 4 + 16];
		const scursor = h("div", { class: "cursor-block", style: { width: "1ch", height: "36px" } });
		strip.append(scursor);
		scursor.style.font = "400 30px/1 var(--mono)";
		grid.append(strip);
		const note = h("div", { class: "label abs", style: { fontSize: "18px", color: "#666" } });
		place(note, 160, 930);
		grid.append(note);

		const mq = h("div", { class: "abs fade-edges", style: { left: "0", top: "0", width: px(W), height: px(H) } });
		const PROVIDERS = ["Anthropic", "OpenAI", "Google", "Groq", "xAI", "Mistral", "Cohere", "Hugging Face"];
		const rowA = h("div", { class: "marquee" });
		const rowB = h("div", { class: "marquee alt" });
		for (let k = 0; k < 2; k++) {
			for (const p of PROVIDERS) rowA.append(h("span", { text: p }), h("span", { class: "sep" }));
		}
		const ALT = ["Keys in your OS credential store", "ChatGPT subscription sign-in", "Adjustable thinking", "saber models set"];
		for (let k = 0; k < 4; k++) {
			for (const p of ALT) rowB.append(h("span", { text: p }), h("span", { class: "sep" }));
		}
		rowA.style.top = "470px";
		rowB.style.top = "650px";
		mq.append(rowA, rowB);
		el.append(mq);

		const S = { tiles: 0.35, cmd: [1.25, 2.65], note: 2.75, swap: [3.25, 3.7], exit: [5.4, 5.85] };

		return {
			el,
			range: T.stack,
			render(t) {
				const exit = E.inCubic(seg(t, S.exit[0], S.exit[1]));
				op(el, seg(t, 0, 0.2) * (1 - exit));

				const swap = E.inOutCubic(seg(t, S.swap[0], S.swap[1]));
				renderLabel(labA, t, 0.05, 31);
				renderLabel(labB, t, S.swap[0] + 0.2, 37);
				op(labA, 1 - swap);
				op(labB, seg(t, S.swap[0] + 0.15, S.swap[0] + 0.3));
				renderRise(headAW, t, 0.1);
				renderRise(headBW, t, S.swap[0] + 0.15);
				op(headA, 1 - seg(t, S.swap[0], S.swap[0] + 0.2));
				tf(headA, `translateY(${(-swap * 40).toFixed(2)}px)`);
				show(headB, t >= S.swap[0]);

				op(grid, 1 - swap);
				tf(grid, `translateY(${(-swap * 70).toFixed(2)}px)`);
				const n = seg(t, S.cmd[0], S.cmd[1]) * cmd.length;
				cmd.show(n);
				tiles.forEach((tile, i) => {
					const p = E.outCubic(seg(t, S.tiles + i * 0.07, S.tiles + 0.5 + i * 0.07));
					op(tile.node, p);
					tf(tile.node, `translateY(${((1 - p) * 26).toFixed(2)}px)`);
					let lit = 0;
					if (i === 4) lit = n >= litAt[0] ? 1 : 0;
					if (i === 5) lit = n >= litAt[1] ? 1 : 0;
					tile.led.style.background = lit ? "#ffffff" : "#2c2c2c";
					op(tile.glow, lit * 0.85);
				});
				op(strip, E.outCubic(seg(t, S.cmd[0] - 0.3, S.cmd[0])));
				const typing = t >= S.cmd[0] - 0.3 && t <= S.cmd[1] + 0.3;
				scursor.style.position = "absolute";
				place(scursor, 36 + Math.floor(n) * 18.36, 29);
				op(scursor, typing && (t <= S.cmd[1] || Math.floor(t * 2.4) % 2 === 0) ? 1 : 0);
				scramble(note, "CSV + Parquet joined in one query via DuckDB", t, S.note, 41, 0.012, 0.15);

				const mqIn = E.outCubic(seg(t, S.swap[0] + 0.2, S.swap[1] + 0.3));
				op(mq, mqIn);
				tf(rowA, `translateX(${(-(t - S.swap[0]) * 320 - 60).toFixed(2)}px)`);
				tf(rowB, `translateX(${((t - S.swap[0]) * 150 - 1400).toFixed(2)}px)`);
			},
		};
	}

	function sceneMore() {
		const el = h("div", { class: "scene" });
		const lab = label("Beyond one question");
		place(lab, 160, 128);
		const head = h("div", { class: "headline abs" });
		place(head, 154, 170);
		const headW = riseWords(head, "Fits your workflow.");
		el.append(lab, head);
		const CARDS = [
			["01", "KNOWLEDGE BASE", "Teach it your KPIs.", [["$ ", "t-p"], ['saber knowledge add "Revenue KPI" "Shipped only"', ""]]],
			["02", "THREADS", "Pick up where you left off.", [["$ ", "t-p"], ["saber threads resume <id>", ""]]],
			["03", "PYTHON SDK", "Build it into your app.", [["", ""], ['await saber.query("Top 5 customers by revenue")', ""]]],
			["04", "MCP SERVER", "Read-only tools for coding agents.", [["$ ", "t-p"], ["saber mcp -d analytics", ""]]],
		];
		const cards = CARDS.map(([ix, lb, title, cm], i) => {
			const cmEl = h("div", { class: "cm" });
			const node = h("div", { class: "card" }, h("div", { class: "ix", text: ix }), h("div", { class: "lb", text: lb }), h("div", { class: "tt", text: title }), cmEl, h("div", { class: "glow" }));
			place(node, 160 + (i % 2) * 816, 330 + Math.floor(i / 2) * 268);
			el.append(node);
			return { node, typed: new Typed(cmEl, cm), glow: node.querySelector(".glow") };
		});
		const S = { cards: 0.35, type0: 0.95, typeDur: 0.55, gap: 0.2, exit: [4.95, 5.4] };

		return {
			el,
			range: T.more,
			render(t) {
				const exit = E.inCubic(seg(t, S.exit[0], S.exit[1]));
				op(el, seg(t, 0, 0.2) * (1 - exit));
				tf(el, exit > 0 ? `scale(${(1 - exit * 0.03).toFixed(4)})` : "none");
				renderLabel(lab, t, 0.05, 51);
				renderRise(headW, t, 0.1);
				cards.forEach((c, i) => {
					const p = E.outCubic(seg(t, S.cards + i * 0.08, S.cards + 0.55 + i * 0.08));
					op(c.node, p);
					tf(c.node, `translateY(${((1 - p) * 30).toFixed(2)}px)`);
					const a = S.type0 + i * (S.typeDur + S.gap);
					c.typed.show(seg(t, a, a + S.typeDur) * c.typed.length);
					const pulse = seg(t, a + S.typeDur, a + S.typeDur + 0.08) * (1 - seg(t, a + S.typeDur + 0.1, a + S.typeDur + 0.7));
					op(c.glow, pulse * 0.9);
				});
			},
		};
	}

	function sceneOutro(grid) {
		const el = h("div", { class: "scene" });
		const ctx = fxCanvas(el);
		const pitch = 12;
		const lw = grid.cols * pitch;
		const lh = grid.rows * pitch;
		const lx = Math.round((W - lw) / 2);
		const ly = 196;
		const tag = h("div", { class: "tagline abs", style: { left: "0", width: px(W), textAlign: "center" }, text: "Ask questions about your data in plain English." });
		place(tag, 0, ly + lh + 66);
		const TEXT = [
			["$ ", "t-p"],
			["uv tool install sqlsaber", ""],
		];
		const box = h("div", { class: "install" });
		const boxW = 26 * 22.032 + 88;
		Object.assign(box.style, { left: px(Math.round((W - boxW) / 2)), top: px(ly + lh + 170), width: px(Math.round(boxW)), padding: "0 44px" });
		const typed = new Typed(box, TEXT);
		const bcursor = h("div", { class: "cursor-block" });
		box.append(bcursor);
		const foot = h("div", { class: "footer-line abs", style: { left: "0", width: px(W), textAlign: "center" } });
		foot.innerHTML = 'sqlsaber.com<span class="dotsep"></span><span class="meta">OPEN SOURCE · APACHE-2.0</span>';
		place(foot, 0, ly + lh + 330);
		el.append(tag, box, foot);
		const S = { on: 0.15, tag: 0.95, box: 1.3, type: [1.45, 2.1], foot: 2.5, off: 5.0 };

		return {
			el,
			range: T.outro,
			render(t) {
				ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
				ctx.clearRect(0, 0, W, H);
				const end = E.inOutCubic(seg(t, S.off + 0.15, S.off + 0.75));
				drawLockup(ctx, grid, { x: lx, y: ly, pitch, t, panel: 0.0, on: S.on, span: 0.7, off: S.off, offSpan: 0.55, alpha: 1 });
				op(tag, E.outCubic(seg(t, S.tag, S.tag + 0.5)) * (1 - end));
				tf(tag, `translateY(${((1 - E.outCubic(seg(t, S.tag, S.tag + 0.5))) * 16).toFixed(2)}px)`);
				const bp = E.outCubic(seg(t, S.box, S.box + 0.4));
				op(box, bp * (1 - end));
				tf(box, `translateY(${((1 - bp) * 18).toFixed(2)}px)`);
				const n = seg(t, S.type[0], S.type[1]) * typed.length;
				typed.show(n);
				bcursor.style.position = "absolute";
				place(bcursor, 44 + Math.floor(n) * 22.032, 26);
				bcursor.style.width = "20px";
				bcursor.style.height = "40px";
				op(bcursor, t < S.type[1] || Math.floor((t - S.type[1]) * 2.2) % 2 === 0 ? 1 : 0);
				const fp = E.outCubic(seg(t, S.foot, S.foot + 0.5));
				op(foot, fp * (1 - end));
				tf(foot, `translateY(${((1 - fp) * 14).toFixed(2)}px)`);
			},
		};
	}

	// ------------------------------------------------------------- timeline

	const scenes = [];

	function seek(t) {
		t = clamp(t, 0, DURATION - 1e-6);
		for (const s of scenes) {
			const [a, b] = s.range;
			const on = t >= a && t < b;
			const d = on ? "block" : "none";
			if (s.el._d !== d) {
				s.el._d = d;
				s.el.style.display = d;
			}
			if (on) s.render(t - a, t);
		}
		op(bgGrid, inOut(t, 3.1, 3.8, DURATION - 1.0, DURATION - 0.3));
	}

	async function init() {
		const params = new URLSearchParams(location.search);
		DPR = window.devicePixelRatio || 1;
		await Promise.all([
			document.fonts.load('700 100px "Doto"'),
			document.fonts.load('300 100px "Space Grotesk"'),
			document.fonts.load('700 100px "Space Grotesk"'),
			document.fonts.load('400 100px "Space Mono"'),
			document.fonts.load('700 100px "Space Mono"'),
		]);
		await document.fonts.ready;

		const grid = buildLockupGrid();
		const intro = sceneIntro(grid);
		const ask = sceneAsk();
		const demo = sceneDemo();
		const safe = sceneSafe();
		const stack = sceneStack();
		const more = sceneMore();
		const outro = sceneOutro(grid);
		scenes.push(intro, ask, demo, safe, stack, more, outro);
		for (const s of scenes) stage.append(s.el);
		ask.el.style.zIndex = "3";
		demo.el.style.zIndex = "2";

		// Measure final layouts once (the morph targets and scroll plan need them).
		for (const s of scenes) s.el.style.display = "block";
		const targets = demo.measure();
		ask.measure(targets);
		safe.measure();
		for (const s of scenes) s.el.style.display = "none";

		window.__meta = { duration: DURATION, fps: FPS, width: W, height: H };
		window.__seek = seek;

		if (params.has("render")) {
			seek(0);
			return window.__meta;
		}
		// Real-time preview.
		const fit = () => {
			const s = Math.min(innerWidth / W, innerHeight / H);
			stage.style.transform = `scale(${s})`;
		};
		fit();
		addEventListener("resize", fit);
		if (params.has("t")) {
			seek(Number(params.get("t")));
			return window.__meta;
		}
		let cur = 0;
		let paused = false;
		let last = performance.now();
		addEventListener("keydown", (e) => {
			if (e.code === "Space") paused = !paused;
			if (e.code === "ArrowRight") cur = Math.min(DURATION - 0.01, cur + 1);
			if (e.code === "ArrowLeft") cur = Math.max(0, cur - 1);
		});
		const loop = (now) => {
			if (!paused) cur = (cur + (now - last) / 1000) % DURATION;
			last = now;
			seek(cur);
			requestAnimationFrame(loop);
		};
		requestAnimationFrame(loop);
		return window.__meta;
	}

	window.__ready = init();
})();
