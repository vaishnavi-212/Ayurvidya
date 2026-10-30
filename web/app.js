// Ayurvidya reading room.
// #/            contents (continue + the chapter's ten sections + how the 7 steps work)
// #/verses      every verse, by section
// #/v/<id>      one verse: palm-leaf text (tap a word) + the 7 steps as one reading
// #/ask[/<id>]  ask a doubt (answer streams in, sources as footnotes)
// #/graph       the five Vayus (when the knowledge graph exists)
// #/notes       my Excel notes (when student notes are enabled)
const CHAPTER = "su.ni.1";
const STEPS = {
  1: { deva: "सम्प्रदानम्", en: "Sampradanam", why: "How the verse divides into its four padas." },
  2: { deva: "पदविभागः", en: "Padavibhaga", why: "Every word separated out, with its grammatical form." },
  3: { deva: "अन्वयः", en: "Anvaya", why: "The same words in natural prose order." },
  4: { deva: "अन्वयार्थः", en: "Anvayartha", why: "The meaning of each word, then of the whole verse." },
  5: { deva: "भावार्थः", en: "Bhavartha", why: "What the Acharya intends to say, in brief." },
  6: { deva: "पदकृत्यम्", en: "Padakrityam", why: "Each word's root, synonyms and grammar." },
  7: { deva: "ध्वनितार्थः", en: "Dhvanitartha", why: "The meaning between the lines." },
};
const SECTIONS = [
  { from: 1, to: 6, deva: "उपोद्घातः", en: "Opening and the greatness of Vayu" },
  { from: 7, to: 10, deva: "वायुस्वरूपम्", en: "Nature, qualities and functions of Vayu" },
  { from: 11, to: 21, deva: "पञ्चवायवः", en: "The five Vayus: seats, actions, disorders" },
  { from: 22, to: 31, deva: "स्थानगतवातः", en: "Vayu lodged in organs and tissues" },
  { from: 32, to: 39, deva: "आवरणम्", en: "Vata joined with or covered by other doshas" },
  { from: 40, to: 50, deva: "वातरक्तम्", en: "Vatarakta: causes, signs and prognosis" },
  { from: 51, to: 59, deva: "आक्षेपकः", en: "Convulsions, spasms and rigidity" },
  { from: 60, to: 67, deva: "पक्षाघातादयः", en: "Hemiplegia, Apatantraka, stiff neck" },
  { from: 68, to: 73, deva: "अर्दितम्", en: "Facial palsy" },
  { from: 74, to: 91, deva: "अन्ये वातव्याधयः", en: "Disorders of limbs, senses and abdomen" },
];
const DIGITS = "०१२३४५६७८९";
const LOTUS = '<svg viewBox="0 0 54 18" aria-hidden="true"><path d="M27 2c3 4 3 9 0 14-3-5-3-10 0-14zM27 16c-4-1-8-4-10-9 5 0 8 3 10 9zm0 0c4-1 8-4 10-9-5 0-8 3-10 9zM2 12h12M40 12h12"/></svg>';

const $ = s => document.querySelector(s);
const view = $("#view");
const cache = { list: null, verse: {}, graph: undefined };
const chats = {};
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
const store = {
  get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};
const session = {
  get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { v == null ? sessionStorage.removeItem(k) : sessionStorage.setItem(k, v); } catch { /* ignore */ } },
};

// ---------------------------------------------------------------- helpers
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const dn = n => String(n).replace(/\d/g, d => DIGITS[d]);
const first = id => parseInt(id.split(".").pop(), 10);
const clean = t => t.replace(/\s*[।॥|]+\s*[०-९0-9\-]+\s*[।॥|]+/g, "").replace(/[ \t]+([।॥])/g, " $1").trim();
const sectionOf = n => SECTIONS.find(s => n >= s.from && n <= s.to) || SECTIONS[SECTIONS.length - 1];
const devaSpans = h => h.replace(/([ऀ-ॿ][ऀ-ॿ\s\-ऽ।॥]*[ऀ-ॿ।॥]|[ऀ-ॿ])/g, '<span class="deva-in">$1</span>');
// [R1] [S4, G2] -> superscript footnotes; refs = {tag: {id,label}} when known
function fnotes(html, refs, order) {
  return html.replace(/\s*\[((?:[RSGN]\d+)(?:\s*,\s*[RSGN]?\d+)*)\]/g, (_, g) => {
    let letter = "R";
    return g.split(/\s*,\s*/).map(p => {
      if (/^[A-Z]/.test(p)) letter = p[0]; else p = letter + p;
      if (order && !order.includes(p)) order.push(p);
      const n = order ? order.indexOf(p) + 1 : p;
      const r = refs && refs[p];
      return `<sup class="fn" data-src="${esc(r ? r.id : "")}" title="${esc(r ? r.label : p)}">${n}</sup>`;
    }).join("");
  });
}
const fmt = (s, refs, order) => fnotes(devaSpans(esc(s)), refs, order);
const seen = () => new Set(JSON.parse(store.get("ayv.seen", "[]")));
function markSeen(id) { const s = seen(); s.add(id); store.set("ayv.seen", JSON.stringify([...s])); store.set("ayv.last", id); }

async function api(path, opts = {}) {
  const tok = session.get("ayv.student");
  opts.headers = Object.assign({}, opts.headers || {}, tok ? { "X-Student": tok } : {});
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `Server error ${r.status}.`;
    try { const j = await r.json(); if (typeof j.detail === "string") msg = j.detail; } catch { /* not JSON */ }
    const e = new Error(msg); e.status = r.status; throw e;
  }
  return r.json();
}
const getList = async () => (cache.list = cache.list || await api(`/chapters/${CHAPTER}/shlokas`));
const getVerse = async id => (cache.verse[id] = cache.verse[id] || await api(`/shlokas/${encodeURIComponent(id)}`));

function page(nav, title) {
  document.querySelectorAll("[data-nav]").forEach(a => a.classList.toggle("on", a.dataset.nav === nav));
  document.title = title ? `${title} · Ayurvidya` : "Ayurvidya";
}
const loading = (n = 4) => { view.innerHTML = Array.from({ length: n }, (_, i) => `<div class="sk" style="width:${90 - i * 12}%"></div>`).join(""); };
function failed(e) {
  view.innerHTML = `<div class="panel"><h2>The server could not be reached</h2>
    <p class="muted">${esc(e.message)} If the site was asleep, it wakes in about a minute. Then try again.</p>
    <button class="btn" onclick="route()">Try again</button></div>`;
}
const ornament = () => `<div class="ornament">${LOTUS}</div>`;

// ---------------------------------------------------------------- contents
async function showHome() {
  page("home", "");
  loading();
  let list;
  try { list = await getList(); } catch (e) { return failed(e); }
  const s = seen(), last = list.find(v => v.id === store.get("ayv.last", "")) || null, start = last || list[0];
  view.innerHTML = `
    <section class="title-page">
      <div class="kicker">Sushruta Samhita · Nidana Sthana · Chapter 1</div>
      <div class="deva">वातव्याधिनिदानम्</div>
      <p>The diagnosis of Vata disorders. ${list.length} verses, each opened in seven steps, from the words on the leaf to the meaning between the lines.</p>
    </section>
    ${ornament()}
    <div class="home">
      <section>
        <a class="bookmark" href="#/v/${start.id}">
          <span class="kicker">${last ? "Continue where you stopped" : "Begin with the first verse"}</span>
          <span class="deva">${esc(clean(start.text))}</span>
          <span class="muted">Verse ${dn(start.number)} · ${esc(sectionOf(first(start.id)).en)}</span>
        </a>
        <h2 style="margin:28px 0 8px">Contents</h2>
        <ol class="toc">${SECTIONS.map((sec, i) => {
          const vs = list.filter(v => first(v.id) >= sec.from && first(v.id) <= sec.to);
          const d = vs.filter(v => s.has(v.id)).length;
          return `<li><a href="#/verses" data-sec="${i}"><span class="n">${dn(i + 1)}</span>
            <span class="t"><span class="deva">${sec.deva}</span><small>${esc(sec.en)}</small></span>
            <span class="r">${dn(sec.from)}–${dn(sec.to)}${d ? `<span class="done">${d} of ${vs.length} read</span>` : ""}</span></a></li>`;
        }).join("")}</ol>
      </section>
      <aside class="panel">
        <h3>How every verse is studied</h3>
        <p class="muted" style="margin:4px 0 10px">Read the steps in order. Each one builds on the one before.</p>
        <ol class="steps-key">${Object.entries(STEPS).map(([n, st]) => `<li><span class="n">${dn(n)}</span>
          <span><b>${st.en}</b> <span class="deva">${st.deva}</span><small>${st.why}</small></span></li>`).join("")}</ol>
        <p style="margin:14px 0 0"><a class="btn line" href="#/ask">Ask a doubt</a></p>
      </aside>
    </div>`;
  view.querySelector(".toc").addEventListener("click", e => { const a = e.target.closest("[data-sec]"); if (a) store.set("ayv.sec", a.dataset.sec); });
}

// ---------------------------------------------------------------- all verses
async function showVerses() {
  page("verses", "Verses");
  loading(8);
  let list;
  try { list = await getList(); } catch (e) { return failed(e); }
  let sec = store.get("ayv.sec", "all");
  view.innerHTML = `<h1>Verses</h1><p class="muted">Choose a section, or type a verse number at the top.</p>
    <div class="chips" id="chips"><button class="chip" data-sec="all">All ${dn(list.length)}</button>
      ${SECTIONS.map((x, i) => `<button class="chip" data-sec="${i}"><span class="deva">${x.deva}</span></button>`).join("")}</div>
    <ul class="vlist" id="vl"></ul>`;
  const s = seen();
  const draw = () => {
    store.set("ayv.sec", sec);
    view.querySelectorAll(".chip").forEach(c => c.classList.toggle("on", c.dataset.sec === String(sec)));
    const S = SECTIONS[sec];
    const rows = list.filter(v => sec === "all" || (first(v.id) >= S.from && first(v.id) <= S.to));
    $("#vl").innerHTML = rows.map(v => `<li class="${s.has(v.id) ? "seen" : ""}"><a href="#/v/${esc(v.id)}">
      <span class="n">${dn(v.number)}</span><span><span class="deva">${esc(clean(v.text))}</span>
      ${v.topic ? `<small>${esc(v.topic)}</small>` : ""}</span></a></li>`).join("") || '<li class="empty">No verses here.</li>';
  };
  $("#chips").addEventListener("click", e => { const c = e.target.closest(".chip"); if (c) { sec = c.dataset.sec; draw(); } });
  draw();
}

// ---------------------------------------------------------------- step renderers (fall back to plain text)
function parse2(out) { return out.split("\n").map(l => l.match(/^\s*\d+\.\s*(.+?)\s+—\s+(.+?)(?:\s+—\s+(.+))?$/)).filter(Boolean); }
function parse4(out) {
  const [head, ...rest] = out.trim().split(/\n\s*\n/);
  const pairs = head.split(/\s*\|\s*|\n/).map(p => p.match(/^\s*(?:हे\s+)?([^=]+?)\s*=\s*(.+)$/)).filter(Boolean);
  return { pairs, sentence: rest.join("\n\n").replace(/^\s*Sentence meaning\s*:\s*/i, "").trim() };
}
function parse6(out) {
  return out.split(/^\s*पद\s*:\s*/m).slice(1).map(b => {
    const lines = b.trim().split("\n"), word = lines.shift().trim(), fields = [];
    for (const l of lines) for (const part of l.split(/\s+\|\s+/)) {
      const m = part.match(/^\s*([^:]{1,24}):\s*(.+)$/);
      if (m) fields.push([m[1].trim(), m[2].trim()]); else if (fields.length) fields[fields.length - 1][1] += " " + part.trim();
    }
    return { word, fields };
  });
}
const render = {
  1(out, r) {
    const rows = out.split("\n").map(l => l.match(/^\s*\S+\s*पाद\s*[-–:]\s*(.+)$/)).filter(Boolean);
    if (rows.length < 2) return null;
    return `<div class="padas">${rows.map((m, i) => `<div class="pada"><span>Pada ${i + 1}</span><div>${fmt(m[1], r)}</div></div>`).join("")}</div>`;
  },
  2(out, r) {
    const rows = parse2(out);
    if (rows.length < 2) return null;
    return `<table class="words"><tbody>${rows.map(m => `<tr><td>${esc(m[1])}</td><td><span class="tag">${esc(m[2])}</span>${m[3] ? fmt(m[3], r) : ""}</td></tr>`).join("")}</tbody></table>`;
  },
  3(out, r) { return `<p class="anvaya">${fmt(out.trim(), r)}</p>`; },
  4(out, r) {
    const { pairs, sentence } = parse4(out);
    if (pairs.length < 2) return null;
    return `<table class="words"><tbody>${pairs.map(m => `<tr><td>${esc(m[1])}</td><td>${fmt(m[2], r)}</td></tr>`).join("")}</tbody></table>
      ${sentence ? `<div class="sense"><b>The whole verse means</b>${fmt(sentence, r)}</div>` : ""}`;
  },
  5(out, r) { return `<div class="prose">${out.trim().split(/\n\s*\n|\n/).map(p => `<p>${fmt(p, r)}</p>`).join("")}</div>`; },
  6(out, r) {
    const blocks = parse6(out);
    if (!blocks.length) return null;
    return blocks.map(b => `<details class="word"><summary><span class="deva">${esc(b.word)}</span>
      <span class="muted">${esc(((b.fields.find(f => /grammar/i.test(f[0])) || [])[1] || "").split(/[,—(]/)[0])}</span></summary>
      <dl>${b.fields.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${fmt(v, r)}</dd>`).join("")}</dl></details>`).join("");
  },
  7(out, r) { return render[5](out, r); },
};
function stepHTML(n, d) {
  if (!d) return '<p class="empty">Not written for this verse yet.</p>';
  try { return render[n](d.output, d.refs) || `<div class="raw">${fmt(d.output, d.refs)}</div>`; }
  catch { return `<div class="raw">${fmt(d.output, d.refs)}</div>`; }
}

// ---------------------------------------------------------------- tap a word
const norm = w => w.replace(/[ंःँ्ऽ।॥\-\s]/g, "");
function wordInfo(token, steps) {
  const t = norm(token), out = [];
  if (!t) return out;
  const hit = w => { const n = norm(w); return n === t || (n.length > 1 && t.length > 1 && (t.includes(n.slice(0, Math.max(2, n.length - 1))) || n.includes(t))); };
  const s2 = steps[2] ? parse2(steps[2].output) : [], s4 = steps[4] ? parse4(steps[4].output).pairs : [], s6 = steps[6] ? parse6(steps[6].output) : [];
  for (const m of s2) if (hit(m[1])) {
    const mean = s4.find(p => norm(p[1]) === norm(m[1]) || hit(p[1]));
    const six = s6.find(b => norm(b.word) === norm(m[1]));
    const root = six && (six.fields.find(f => /धातु|root/i.test(f[0])) || [])[1];
    out.push({ word: m[1], gram: [m[2], m[3]].filter(Boolean).join(" — "), mean: mean ? mean[2] : "", root: root || "" });
    if (out.length === 3) break;
  }
  return out;
}
let pop = null;
function closePop() { pop?.remove(); pop = null; document.querySelectorAll(".w.on").forEach(x => x.classList.remove("on")); }
function openPop(el, steps) {
  closePop();
  el.classList.add("on");
  const info = wordInfo(el.textContent, steps);
  pop = document.createElement("div");
  pop.className = "pop";
  pop.setAttribute("role", "dialog");
  pop.innerHTML = info.length ? info.map(i => `<div class="deva">${esc(i.word)}</div><dl>
      ${i.gram ? `<dt>Form</dt><dd>${fmt(i.gram)}</dd>` : ""}${i.mean ? `<dt>Meaning</dt><dd>${fmt(i.mean)}</dd>` : ""}
      ${i.root ? `<dt>Root</dt><dd>${fmt(i.root)}</dd>` : ""}</dl>`).join('<hr style="border:0;border-top:1px dotted var(--rule)">')
    : `<div class="deva">${esc(el.textContent)}</div><p class="muted" style="margin:6px 0 0">This form is not listed by itself. Look in Padavibhaga (step 2) below.</p>`;
  document.body.appendChild(pop);
  const r = el.getBoundingClientRect(), w = pop.offsetWidth;
  pop.style.left = Math.max(8, Math.min(scrollX + r.left + r.width / 2 - w / 2, scrollX + innerWidth - w - 8)) + "px";
  pop.style.top = scrollY + r.bottom + 8 + "px";
}
document.addEventListener("click", e => { if (pop && !e.target.closest(".pop") && !e.target.closest(".w")) closePop(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") { closePop(); closeSheet(); } });

// ---------------------------------------------------------------- footnote sheet
function openSheet(html) { $("#sheetBody").innerHTML = html; $("#sheet").classList.add("open"); $("#sheet").setAttribute("aria-hidden", "false"); $("#sheetX").focus(); }
function closeSheet() { $("#sheet").classList.remove("open"); $("#sheet").setAttribute("aria-hidden", "true"); }
$("#sheetX").onclick = closeSheet;
$("#sheet").addEventListener("click", e => { if (e.target.id === "sheet") closeSheet(); });
document.addEventListener("click", async e => {
  const f = e.target.closest("sup.fn");
  if (!f) return;
  const id = f.dataset.src;
  if (!id) { openSheet(`<h3 id="sheetTitle">Source ${esc(f.textContent)}</h3><p class="muted">${esc(f.title)}: the passage used when this was written.</p>`); return; }
  openSheet(`<h3 id="sheetTitle">${esc(f.title)}</h3><div class="sk"></div><div class="sk"></div>`);
  try {
    if (id.startsWith("note:")) { $("#sheetBody").innerHTML = `<h3 id="sheetTitle">Your notes</h3><p class="muted">From the Excel sheet you uploaded. Open My notes to see it.</p>`; return; }
    const s = await api(`/source/${encodeURIComponent(id)}`);
    $("#sheetBody").innerHTML = `<div class="kicker">${esc(s.type || "study step")}</div><h3 id="sheetTitle" style="margin:4px 0 12px">${esc(s.label)}</h3>
      <div class="raw" style="line-height:1.8">${devaSpans(esc(s.text))}</div>`;
  } catch (err) { $("#sheetBody").innerHTML = `<p class="err">${esc(err.message)}</p>`; }
});

// ---------------------------------------------------------------- one verse
async function showVerse(id) {
  page("verses", `Verse ${dn(first(id))}`);
  loading(6);
  let list, data;
  try { [list, data] = await Promise.all([getList(), getVerse(id)]); } catch (e) { return failed(e); }
  const i = list.findIndex(v => v.id === id), prev = list[i - 1], next = list[i + 1];
  const v = data.shloka, sec = sectionOf(first(id)), steps = data.steps;
  page("verses", `Verse ${dn(v.number)}`);
  markSeen(id);
  const verseHTML = esc(clean(v.text)).replace(/[ \t]+([।॥])/g, " $1").split("\n").map(line =>
    line.split(/(\s+)/).map(t => /[ऀ-ॿ]/.test(t) && !/^[।॥]+$/.test(t) ? `<span class="w" tabindex="0" role="button">${t}</span>` : t).join("")).join("\n");
  view.innerHTML = `
    <div class="study">
      <nav class="rail" id="rail" aria-label="The seven steps">${Object.entries(STEPS).map(([n, s]) =>
        `<a href="#s${n}" data-s="${n}"><span class="deva">${dn(n)}</span>${s.en}</a>`).join("")}</nav>
      <article class="reading">
        <div class="verse-meta"><span class="kicker">${esc(sec.en)} · <b>verse ${dn(v.number)}</b></span>
          <a href="#/ask/${esc(id)}">Ask about this verse</a></div>
        <div class="leaf"><div class="num">श्लोकः ${dn(v.number)}</div><div class="verse" id="verse">${verseHTML}</div></div>
        ${v.variant ? `<div class="variant">Variant reading (पाठान्तरम्): <span class="deva">${esc(v.variant)}</span></div>` : ""}
        <p class="muted" style="font-size:.92rem;margin:10px 0 0">Tap any word on the leaf for its form and meaning.</p>
        <div id="mynote"></div>
        ${Object.entries(STEPS).map(([n, s]) => `
          <section class="step" id="s${n}">
            <div class="step-h"><span class="num">${dn(n)}</span><h2>${s.en}<span class="deva">${s.deva}</span></h2></div>
            <p class="step-why">${s.why}</p>
            <div class="step-body">${stepHTML(+n, steps[n])}</div>
          </section>`).join("")}
        <div class="pager">
          ${prev ? `<a href="#/v/${prev.id}">← Verse ${dn(prev.number)}</a>` : "<span></span>"}
          ${next ? `<a href="#/v/${next.id}">Verse ${dn(next.number)} →</a>` : "<span></span>"}
        </div>
      </article>
    </div>`;
  const verseEl = $("#verse");
  verseEl.addEventListener("click", e => { const w = e.target.closest(".w"); if (w) openPop(w, steps); });
  verseEl.addEventListener("keydown", e => { const w = e.target.closest(".w"); if (w && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); openPop(w, steps); } });
  // rail follows the reading
  const links = [...document.querySelectorAll("#rail a")];
  const io = new IntersectionObserver(es => es.forEach(en => {
    if (en.isIntersecting) links.forEach(a => a.classList.toggle("on", a.dataset.s === en.target.id.slice(1)));
  }), { rootMargin: "-30% 0px -60% 0px" });
  document.querySelectorAll(".step").forEach(s => io.observe(s));
  $("#rail").addEventListener("click", e => {
    const a = e.target.closest("a"); if (!a) return;
    e.preventDefault();
    document.getElementById(`s${a.dataset.s}`).scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
  });
  // keyboard: j / k between steps, arrows between verses
  onKeys = e => {
    if (e.target.closest("input, textarea")) return;
    if (e.key === "ArrowRight" && next) location.hash = `#/v/${next.id}`;
    if (e.key === "ArrowLeft" && prev) location.hash = `#/v/${prev.id}`;
    if (e.key === "j" || e.key === "k") {
      const cur = links.findIndex(a => a.classList.contains("on"));
      const to = Math.max(0, Math.min(6, (cur < 0 ? 0 : cur) + (e.key === "j" ? 1 : -1)));
      links[to].click();
    }
  };
  if (session.get("ayv.student")) {
    api(`/notes/verse/${encodeURIComponent(id)}`).then(ns => {
      if (ns.length) $("#mynote").innerHTML = ns.map(n => `<div class="mynote"><b>Your ${esc(n.kind)}${n.word ? ` of ${esc(n.word)}` : ""}:</b> ${fmt(n.text)}</div>`).join("");
    }).catch(() => {});
  }
}
let onKeys = null;
document.addEventListener("keydown", e => onKeys && onKeys(e));

// ---------------------------------------------------------------- ask
async function streamChat(body, onToken) {
  const tok = session.get("ayv.student");
  const r = await fetch("/chat/stream", { method: "POST", headers: Object.assign({ "Content-Type": "application/json" }, tok ? { "X-Student": tok } : {}), body: JSON.stringify(body) });
  if (!r.ok || !r.body) {                                  // old server: fall back to the plain endpoint
    if (r.status === 404 || r.status === 405) return api("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let msg = `Server error ${r.status}.`; try { msg = (await r.json()).detail || msg; } catch { /* */ } throw new Error(msg);
  }
  const reader = r.body.getReader(), dec = new TextDecoder();
  let buf = "", done = null;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (end) break;
    buf += dec.decode(value, { stream: true });
    let k;
    while ((k = buf.indexOf("\n\n")) >= 0) {
      const line = buf.slice(0, k).trim(); buf = buf.slice(k + 2);
      if (!line.startsWith("data:")) continue;
      const ev = JSON.parse(line.slice(5));
      if (ev.type === "token") onToken(ev.text);
      else if (ev.type === "done") done = ev;
      else if (ev.type === "error") throw new Error(ev.message);
    }
  }
  if (!done) throw new Error("The answer was cut off. Please ask again.");
  return done;
}
function answerHTML(a) {
  const order = [];
  let html = a.answer.split(/\n+/).map(p => {
    let h = fmt(p, a.tags || {}, order);
    for (const w of a.weak || []) {
      const wf = fmt(w, a.tags || {}, []);
      if (wf && h.includes(wf.replace(/<sup[^>]*>\d+<\/sup>/g, ""))) h = h.replace(wf.replace(/<sup[^>]*>\d+<\/sup>/g, ""), m => `<span class="weak" title="The cited source may not say this. Check it.">${m}</span>`);
    }
    return `<p>${h}</p>`;
  }).join("");
  if (order.length) html += `<div class="fns">${order.map((t, i) => { const r = (a.tags || {})[t]; return `<div><b>${i + 1}</b> ${esc(r ? r.label : t)}</div>`; }).join("")}</div>`;
  if ((a.weak || []).length) html += `<div class="note">Dotted underline: the cited source may not fully support that sentence. Open the footnote and check.</div>`;
  return html;
}
async function showAsk(id) {
  page("ask", "Ask");
  let data = null;
  if (id) { loading(3); try { data = await getVerse(id); } catch (e) { return failed(e); } store.set("ayv.last", id); }
  const key = id || "_", history = chats[key] = chats[key] || [];
  const words = data ? parse2(data.steps[2]?.output || "").map(m => m[1]) : [];
  const ideas = data ? [words[0] && `What does ${words[0]} mean here?`, "Explain the grammar of this verse simply",
    "Why does the Acharya say this?", words[1] && `Which root does ${words[1]} come from?`].filter(Boolean)
    : ["Names of the five types of Vayu", "Which Vayu causes hiccup and breathing trouble?", "What causes Vatarakta?", "What is Pakshaghata (hemiplegia)?"];
  view.innerHTML = `
    <div class="ask">
      <aside>
        ${data ? `<a href="#/v/${id}" style="text-decoration:none;color:inherit"><div class="leaf" style="padding:18px 40px">
          <div class="num">श्लोकः ${dn(data.shloka.number)}</div><div class="verse" style="font-size:1.15rem">${esc(clean(data.shloka.text)).replace(/[ \t]+([।॥])/g, " $1")}</div></div></a>`
          : `<h1>Ask a doubt</h1><p class="muted">About any verse of this chapter. To ask about one verse, open it and choose “Ask about this verse”.</p>`}
        <div class="ideas" id="ideas"><span class="kicker" style="margin-top:12px">Try asking</span>${ideas.map(t => `<button>${esc(t)}</button>`).join("")}</div>
        <p class="muted" style="font-size:.9rem;margin-top:16px">Answers come only from this chapter, its study steps and checked sources, with footnotes. Confirm important points with your teacher.</p>
      </aside>
      <section>
        <div class="thread" id="thread"></div>
        <form class="composer" id="f"><label for="msg" class="sr">Your doubt</label>
          <textarea id="msg" rows="1" placeholder="Type your doubt…"></textarea><button class="btn" id="send">Ask</button></form>
      </section>
    </div>`;
  const thread = $("#thread"), input = $("#msg"), send = $("#send");
  const add = q => {
    const d = document.createElement("div");
    d.innerHTML = `<div class="q">${esc(q)}</div><div class="a"><span class="cursor"></span></div>`;
    thread.appendChild(d);
    return d.querySelector(".a");
  };
  history.forEach(h => { add(h.q).innerHTML = answerHTML(h.a); });
  async function ask(q) {
    const box = add(q);
    box.scrollIntoView({ block: "nearest", behavior: reduced ? "auto" : "smooth" });
    send.disabled = true;
    let text = "";
    try {
      const a = await streamChat({ question: q, shloka_id: id || null, history: history.slice(-3).map(h => [h.q, h.a.answer]) },
        t => { text += t; box.innerHTML = fmt(text) + '<span class="cursor"></span>'; });
      box.innerHTML = answerHTML(a);
      history.push({ q, a });
    } catch (err) { box.innerHTML = `<p class="err">No answer this time. ${esc(err.message)}</p>`; }
    finally { send.disabled = false; input.focus(); }
  }
  $("#ideas").addEventListener("click", e => { const b = e.target.closest("button"); if (b) ask(b.textContent); });
  input.addEventListener("input", () => { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; });
  input.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#f").requestSubmit(); } });
  $("#f").addEventListener("submit", e => { e.preventDefault(); const q = input.value.trim(); if (!q) return; input.value = ""; input.style.height = "auto"; ask(q); });
}

// ---------------------------------------------------------------- Vayu map (knowledge graph)
async function getGraph() {
  if (cache.graph === undefined) { try { cache.graph = await api("/graph"); } catch { cache.graph = null; } }
  return cache.graph;
}
async function showGraph() {
  page("graph", "Vayu map");
  loading(5);
  const g = await getGraph();
  if (!g) { view.innerHTML = '<p class="empty">The Vayu map is not ready yet.</p>'; return; }
  const items = xs => (xs || []).map(x => `${esc(x.text)}${x.verse ? ` <a href="#/v/${esc(x.verse)}">${dn(first(x.verse))}</a>` : ""}`).join("; ") || "—";
  view.innerHTML = `<h1>The five Vayus</h1><p class="muted">Where each Vayu lives, what it does, and the disorders it causes when vitiated. ${g.n_facts} facts, each tied to the exact words of its verse (${g.checked} checked by a scholar so far). Tap a verse number to read it.</p>
    ${ornament()}
    <div class="vayus">${g.vayus.map(x => `<section class="panel vayu"><h3><span class="deva">${esc(x.deva)}</span>${esc(x.name)}</h3><dl>
      <dt>Seat</dt><dd>${items(x.sthana)}</dd><dt>Actions</dt><dd>${items(x.karma)}</dd>
      <dt>Disorders</dt><dd>${items(x.vyadhi)}</dd>${(x.other || []).length ? `<dt>Also</dt><dd>${items(x.other)}</dd>` : ""}</dl></section>`).join("")}</div>`;
}

// ---------------------------------------------------------------- my notes (Excel upload)
async function showNotes() {
  page("notes", "My notes");
  const tok = session.get("ayv.student");
  if (!tok) {
    view.innerHTML = `<h1>My notes</h1><p class="muted">Upload your own translations and notes from an Excel sheet. They are kept for you only, and the tutor will use them when you ask a doubt.</p>
      <form class="form panel" id="login"><label>Class code<input id="code" required autocomplete="off"></label>
      <label>Your name<input id="name" required autocomplete="name"></label><button class="btn">Continue</button><p class="err" id="lerr"></p></form>`;
    $("#login").addEventListener("submit", async e => {
      e.preventDefault();
      try {
        const r = await api("/notes/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ class_code: $("#code").value.trim(), name: $("#name").value.trim() }) });
        session.set("ayv.student", r.token); session.set("ayv.sname", r.name); showNotes();
      } catch (err) { $("#lerr").textContent = err.message; }
    });
    return;
  }
  loading(4);
  let notes;
  try { notes = await api("/notes"); } catch (e) { if (e.status === 401) { session.set("ayv.student", null); return showNotes(); } return failed(e); }
  view.innerHTML = `<h1>My notes</h1><p class="muted">Signed in as ${esc(session.get("ayv.sname") || "")}. <a href="#/notes" id="out">Sign out</a></p>
    <div class="panel"><h3>Upload an Excel sheet</h3>
      <ol class="muted" style="margin:8px 0 12px;padding-left:20px"><li><a href="/notes/template" download>Download the template</a> and fill one row per note.</li>
      <li>Upload it here. Uploading again replaces your earlier rows for the same verse, kind and word.</li></ol>
      <form id="up" class="form"><input type="file" id="file" accept=".xlsx" required><button class="btn">Upload</button></form>
      <div class="report" id="rep"></div>
      <p style="margin-top:12px;display:flex;gap:10px;flex-wrap:wrap"><button class="btn line" id="undo" ${notes.length ? "" : "disabled"}>Undo last upload</button>
      <button class="btn line" id="submit" ${notes.some(n => n.status === "personal") ? "" : "disabled"}>Send to my teacher for checking</button></p></div>
    <h2 style="margin-top:26px">${dn(notes.length)} notes</h2>
    ${notes.length ? `<table class="notes"><thead><tr><th>Verse</th><th>Kind</th><th>Note</th></tr></thead><tbody>${notes.map(n => `<tr>
      <td><a href="#/v/${esc(n.verse_id)}">${dn(n.verse)}</a></td><td>${esc(n.kind)}${n.word ? `<br><span class="deva">${esc(n.word)}</span>` : ""}${n.status !== "personal" ? `<br><small class="muted">${esc(n.status)}</small>` : ""}</td><td>${fmt(n.text)}</td></tr>`).join("")}</tbody></table>`
      : '<p class="empty">Nothing uploaded yet.</p>'}`;
  $("#out").onclick = e => { e.preventDefault(); session.set("ayv.student", null); showNotes(); };
  $("#up").addEventListener("submit", async e => {
    e.preventDefault();
    const fd = new FormData(); fd.append("file", $("#file").files[0]);
    $("#rep").textContent = "Checking your sheet…";
    try {
      const r = await api("/notes/upload", { method: "POST", body: fd });
      $("#rep").innerHTML = `<p><b>${r.ok_rows} rows saved.</b>${r.errors.length ? ` ${r.errors.length} rows need fixing:` : ""}</p>
        ${r.errors.length ? `<ul>${r.errors.slice(0, 30).map(x => `<li>Row ${x.row}: ${esc(x.problem)}</li>`).join("")}</ul>` : ""}`;
      if (r.ok_rows) setTimeout(showNotes, 1400);
    } catch (err) { $("#rep").innerHTML = `<p class="err">${esc(err.message)}</p>`; }
  });
  $("#submit").onclick = async () => { try { const r = await api("/notes/submit", { method: "POST" }); $("#rep").innerHTML = `<p>${r.submitted} notes sent to your teacher.</p>`; } catch (err) { $("#rep").innerHTML = `<p class="err">${esc(err.message)}</p>`; } };
  $("#undo").onclick = async () => { try { await api("/notes/undo", { method: "POST" }); showNotes(); } catch (err) { $("#rep").innerHTML = `<p class="err">${esc(err.message)}</p>`; } };
}

// ---------------------------------------------------------------- router + shell
async function route() {
  onKeys = null; closePop(); closeSheet();
  const h = decodeURIComponent(location.hash);
  let m;
  window.scrollTo(0, 0);
  if ((m = h.match(/^#\/v\/([^/]+)/))) return showVerse(m[1]);
  if ((m = h.match(/^#\/ask(?:\/(.+))?$/))) return showAsk(m[1]);
  if ((m = h.match(/^#\/chat\/(.+)$/))) return showAsk(m[1]);       // old links
  if (h === "#/verses") return showVerses();
  if (h === "#/graph") return showGraph();
  if (h === "#/notes") return showNotes();
  return showHome();
}
$("#theme").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = t;
  document.querySelector('meta[name="theme-color"]').content = t === "dark" ? "#1d1610" : "#efe3c8";
  store.set("ayv.theme", t);
};
$("#jump").addEventListener("submit", async e => {
  e.preventDefault();
  const n = $("#jumpIn").value.trim().replace(/[०-९]/g, d => DIGITS.indexOf(d));
  $("#jumpIn").value = "";
  if (!n) return;
  try { const v = (await getList()).find(x => x.number.split("-").includes(n)); if (v) location.hash = `#/v/${v.id}`; } catch (err) { failed(err); }
});
// show the optional pages only when the server has them
getGraph().then(g => { if (g) document.querySelectorAll('[data-nav="graph"]').forEach(a => (a.hidden = false)); });
fetch("/notes/enabled").then(r => r.ok && r.json()).then(j => { if (j && j.enabled) document.querySelectorAll('[data-nav="notes"]').forEach(a => (a.hidden = false)); }).catch(() => {});
window.addEventListener("hashchange", route);
route();

if ("serviceWorker" in navigator && window.isSecureContext) navigator.serviceWorker.register("sw.js").catch(() => {});