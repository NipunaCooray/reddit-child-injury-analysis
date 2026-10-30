// Drives public/index.html in jsdom against an in-memory store, once as the study lead and once as a coder.
// The tutorial and codebook come from the repo's exporter; the "real" post reuses a tutorial post's text.
// Access rules are tested separately against the emulator (tests/rules.test.mjs).
const fs = require("fs"), path = require("path"), os = require("os"), { execFileSync } = require("child_process");
const { JSDOM } = require("jsdom");

const ROOT = path.resolve(__dirname, "..", "..");
const out = fs.mkdtempSync(path.join(os.tmpdir(), "coding-app-"));
execFileSync("python3", [path.join(ROOT, "src/extract/export_coding_app.py"), "--outdir", out], { stdio: "ignore" });
const read = (p) => JSON.parse(fs.readFileSync(p, "utf8"));
const clone = (x) => JSON.parse(JSON.stringify(x));
const codebook = read(path.join(out, "codebook.json"));
const tutorialFiles = fs.readdirSync(path.join(out, "tutorial")).map(f => path.join(out, "tutorial", f));
const keys = Object.fromEntries(tutorialFiles.map(f => read(f)).map(d => [d.id, d]));
const html = fs.readFileSync(path.join(__dirname, "..", "public", "index.html"), "utf8");

function merge(a, b) { const o = { ...a }; for (const [k, v] of Object.entries(b)) o[k] = v && typeof v === "object" && !Array.isArray(v) && a?.[k] && typeof a[k] === "object" && !Array.isArray(a[k]) ? merge(a[k], v) : v; return o; }
class FakeDB {
  constructor() { this.data = new Map(); this.ls = []; }
  doc(p) { const db = this; return {
    async set(d) { db.write(p, clone(d)); },
    async update(d) { db.write(p, merge(db.data.get(p) || {}, clone(d))); },
    async delete() { db.data.delete(p); db.notify(p); },
    async get() { return db.dsnap(p); },
    onSnapshot(cb) { const l = { p, cb }; db.ls.push(l); setTimeout(() => cb(db.dsnap(p)), 0); return () => { db.ls = db.ls.filter(x => x !== l); }; } }; }
  collection(n) { const db = this; return { doc: (id) => db.doc(n + "/" + id), async get() { return db.csnap(n); },
    onSnapshot(cb) { const l = { n, cb }; db.ls.push(l); setTimeout(() => cb(db.csnap(n)), 0); return () => { db.ls = db.ls.filter(x => x !== l); }; } }; }
  async batch(ops) { for (const o of ops) o.op === "delete" ? this.data.delete(o.path) : this.data.set(o.path, clone(o.data)); for (const o of ops) this.notify(o.path); }
  dsnap(p) { const d = this.data.get(p); return { exists: !!d, data: () => clone(d) }; }
  csnap(n) { return { docs: [...this.data].filter(([p]) => p.split("/")[0] === n && p.split("/").length === 2).map(([p, d]) => ({ id: p.split("/")[1], data: () => clone(d) })) }; }
  write(p, d) { this.data.set(p, d); this.notify(p); }
  notify(p) { setTimeout(() => { for (const l of this.ls) { if (l.p === p) l.cb(this.dsnap(p)); if (l.n && p.split("/")[0] === l.n) l.cb(this.csnap(l.n)); } }, 3); }
}

const NAMES = { uLead: "Lead", uCoder: "Coder" }; // what people/ holds in the real store
let fails = 0; const ok = (cond, msg) => { console.log((cond ? "PASS " : "FAIL ") + msg); if (!cond) fails++; };
const sleep = (ms = 40) => new Promise(r => setTimeout(r, ms));

function open(db, person) {
  const saved = [];
  const backend = { db, role: person.role, email: person.email, signOut() {}, download: (filename, data) => saved.push({ filename, data }),
    user: { me: async () => ({ id: person.uid, name: person.name }), profiles: async (ids) => Object.fromEntries([].concat(ids).map(i => [i, { id: i, name: NAMES[i] || "" }])) } };
  const dom = new JSDOM(html, { runScripts: "dangerously", pretendToBeVisual: true, url: "https://human-coding.web.app/",
    beforeParse(w) { w.HC_BACKEND = Promise.resolve(backend); w.Element.prototype.scrollIntoView = () => {}; w.scrollTo = () => {}; w.confirm = () => true; } });
  const w = dom.window, d = w.document;
  w.addEventListener("error", e => console.error("PAGE ERROR", e.message));
  const q = (s) => d.querySelector(s), qa = (s) => [...d.querySelectorAll(s)];
  const btn = (text, root = d) => [...root.querySelectorAll("button")].find(b => b.textContent.trim().startsWith(text));
  const fire = (el, type) => el.dispatchEvent(new w.Event(type, { bubbles: true }));
  async function fillCell(f, c) {
    let cell = q("#cell-" + f); if (!cell || cell.classList.contains("locked")) return;
    cell.querySelector(`button[data-s="${c.status}"]`).click(); await sleep();
    if (c.status === "not_stated") return;
    cell = q("#cell-" + f);
    const sel = cell.querySelector("select"), multi = cell.querySelector(".multi"), txt = cell.querySelector('input[type="text"]');
    if (sel) { sel.value = c.value; fire(sel, "change"); await sleep(); }
    else if (multi) { for (const v of c.value) { const box = [...q("#cell-" + f).querySelectorAll(".multi label")].find(l => l.textContent === v).querySelector("input"); box.checked = true; fire(box, "change"); await sleep(); } }
    else if (txt) { txt.value = c.value; fire(txt, "input"); fire(txt, "change"); await sleep(); }
    const ta = q("#q-" + f); ta.value = c.quote; fire(ta, "input"); fire(ta, "change"); await sleep();
  }
  const fields = codebook.fields.map(x => x.field);
  const codeFromKey = async (key, override = {}) => { for (const f of fields) await fillCell(f, override[f] || key[f]); };
  // visible text only: the app's own script sits in <body>, so body.textContent would match its source
  const seen = () => q("main").textContent + q("header").textContent + q("#notice").textContent;
  return { w, d, q, qa, btn, fire, fillCell, codeFromKey, saved, seen };
}

// Choose files in an <input type=file> (jsdom has no DataTransfer)
async function chooseFiles(app, input, paths) {
  const files = paths.map(p => { const f = new app.w.File([fs.readFileSync(p, "utf8")], path.basename(p), { type: "application/json" }); f.text = async () => fs.readFileSync(p, "utf8"); return f; });
  Object.defineProperty(input, "files", { value: files, configurable: true });
  app.fire(input, "change"); await sleep(120);
}

(async () => {
  const db = new FakeDB();
  const lead = { uid: "uLead", email: "lead@example.org", role: "admin", name: "Lead" };
  const coder = { uid: "uCoder", email: "coder@uni.example.edu", role: "coder", name: "Coder" };
  db.data.set("roles/" + lead.email, { role: "admin" });

  // ---- A. the study lead loads everything through the Admin tab ----
  let a = open(db, lead); await sleep(150);
  ok(!a.q("#tabAdmin").hidden, "lead sees the Admin tab");
  ok(a.seen().includes("No codebook loaded yet"), "empty store points the lead to Admin");
  a.q("#tabAdmin").click(); await sleep();
  a.q("#addEmail").value = "Coder@Uni.Example.edu"; a.q("#addRole").value = "coder"; a.btn("Add").click(); await sleep(60);
  ok(db.data.get("roles/coder@uni.example.edu")?.role === "coder", "adds a person, email lower-cased");
  ok(a.qa("tbody tr").some(tr => tr.textContent.includes("coder@uni.example.edu")), "people table lists the new coder");
  const post = { id: "fake1", title: keys["tutorial-4-scald"].title, selftext: keys["tutorial-4-scald"].selftext, permalink: "", sample: "practice", order: 1 };
  const postFile = path.join(out, "fake1.json"); fs.writeFileSync(postFile, JSON.stringify(post));
  await chooseFiles(a, a.q('input[type="file"][multiple]'), [path.join(out, "codebook.json"), ...tutorialFiles, postFile]);
  const ready = a.q(".banner.info")?.textContent || "";
  ok(/Codebook, schema 1\.5\.0/.test(ready) && /8 tutorial posts/.test(ready) && /1 posts in sample "practice"/.test(ready), "load preview summarises the files");
  a.btn("Load ").click(); await sleep(120);
  ok(db.data.has("config/codebook") && db.data.has("posts/fake1"), "codebook and post loaded");
  ok(db.data.get("tutorial/tutorial-1-stairs") && !("key" in db.data.get("tutorial/tutorial-1-stairs")) && !("label" in db.data.get("tutorial/tutorial-1-stairs")), "tutorial post stored without its key or label");
  ok(db.data.get("tutorial_keys/tutorial-1-stairs")?.label === "Near miss", "answer key stored apart");
  await chooseFiles(a, a.q('input[type="file"][multiple]'), [postFile]);
  ok(/already loaded and will be replaced/.test(a.seen()), "warns before replacing a loaded post");
  a.btn("Cancel").click(); await sleep();

  // ---- B. the coder: tutorial, key opens only after checking ----
  const before = db.ls.length;
  const c = open(db, coder); await sleep(150);
  const coderListens = () => db.ls.slice(before);
  ok(c.q("#tabAdmin").hidden && c.q("#tabReview").hidden, "coder sees neither Admin nor Review");
  ok(c.q("#tabTutorial").getAttribute("aria-pressed") === "true" && c.qa(".plist li").length === 8, "coder opens on the tutorial, 8 posts");
  ok(!coderListens().some(l => l.p?.startsWith("tutorial_keys/") || l.n === "tutorial_keys"), "no answer key requested before checking");
  c.qa(".plist button").find(b => b.textContent.includes(keys["tutorial-7-bath"].title)).click(); await sleep();
  await c.codeFromKey(keys["tutorial-7-bath"].key);
  c.btn("Check against the key").click(); await sleep(150);
  ok(db.data.get("tutorial_codes/tutorial-7-bath--uCoder")?.state === "submitted", "attempt saved under the coder's id");
  ok(coderListens().filter(l => l.p?.startsWith("tutorial_keys/") || l.n === "tutorial_keys").map(l => l.p).join() === "tutorial_keys/tutorial-7-bath", "only that post's key is requested");
  ok(/^25 of 25 fields match the key/.test(c.q(".pane.form .banner")?.textContent || ""), "key coding scores 25 of 25");
  ok(c.q(".section-title").textContent.includes("Submersion near miss"), "label shown after checking");
  ok(!c.q("#coderNote"), "no reviewer note in the tutorial");

  // ---- C. the coder codes the practice post ----
  c.q("#tabCode").click(); await sleep(80);
  ok(c.q(".story h1").textContent === post.title, "Code tab shows the practice post");
  await c.codeFromKey(keys["tutorial-4-scald"].key, { injury_event_check: { status: "inferred", value: "suspected_injury_event", quote: "It went down her arm" } });
  const note = c.q("#coderNote"); note.value = "Unsure if 'went down her arm' is enough for suspected"; c.fire(note, "input"); await sleep(1000);
  ok(db.data.get("codes/fake1")?.note === note.value, "note for the reviewer saves with the coding");
  c.btn("Submit for review").click(); await sleep(80);
  ok(db.data.get("codes/fake1")?.state === "submitted" && db.data.get("codes/fake1").coder === "uCoder", "coding submitted under the coder's id");

  // ---- D. the lead reviews (as reviewer), the coder replies, the lead resolves and exports ----
  a = open(db, lead); await sleep(150);
  a.q("#tabReview").click(); await sleep(80);
  ok(!a.seen().includes("You coded this post") && a.qa(".pane.form .cell").length === 25, "review opens the coder's post with no same-person warning");
  ok(a.q(".coder-note")?.textContent.includes("went down her arm"), "reviewer sees the coder's note");
  const rc = () => a.qa(".pane.form .cell").find(x => x.querySelector(".fname")?.textContent === "injury_event_check");
  a.btn("Flag", rc()).click(); await sleep(60);
  rc().querySelector(".thread textarea").value = "Blistered patch is a stated injury"; a.btn("Send", rc()).click(); await sleep(60);
  ok(db.data.get("reviews/fake1")?.items?.injury_event_check?.verdict === "flag", "flag saved");
  const c2 = open(db, coder); await sleep(150); c2.q("#tabCode").click(); await sleep(80);
  const cc = c2.qa(".pane.form .cell").find(x => x.querySelector(".fname")?.textContent === "injury_event_check");
  cc.querySelector(".thread textarea").value = "Agreed, it is stated"; c2.btn("Send", cc).click(); await sleep(80);
  const th = db.data.get("reviews/fake1").items.injury_event_check.thread;
  ok(th.length === 2 && th[1].role === "coder" && db.data.get("reviews/fake1").items.injury_event_check.verdict === "flag", "coder's reply merges into the thread without touching the flag");
  await sleep(60);
  const resSel = rc().querySelector(".resolution select"); resSel.value = "actual_injury_event"; a.fire(resSel, "change"); await sleep(60);
  rc().querySelector('.resolution button[data-s="stated"]').click(); await sleep(60);
  const rq = rc().querySelector(".resolution textarea"); rq.value = "Blistered patch about the size of a 50c coin on her forearm"; a.fire(rq, "input"); a.fire(rq, "change"); await sleep(60);
  rc().querySelector('.resolution input[type="checkbox"]').click(); await sleep(60);
  a.btn("Agree with").click(); await sleep(60);
  a.btn("Mark post agreed").click(); await sleep(80);
  const rv = db.data.get("reviews/fake1");
  ok(rv.state === "agreed" && rv.items.injury_event_check.thread.length === 2, "review agreed with both thread messages kept");
  a.q("#tabProgress").click(); await sleep(80);
  ok(a.seen().includes("Coder") && a.seen().includes("1 of 8"), "Progress shows the coder's tutorial result by name");
  a.btn("Download agreed coding").click(); await sleep(60);
  const rows = JSON.parse(a.saved.at(-1)?.data || "[]");
  ok(rows.length === 1 && rows[0].fields.injury_event_check.value === "actual_injury_event", "export applies the resolved flag");
  ok(rows[0]?.coder_note?.includes("went down her arm"), "export carries the coder's note");
  a.q("#tabAdmin").click(); await sleep(60);
  a.btn("Download a full backup").click(); await sleep(120);
  const bk = JSON.parse(a.saved.at(-1)?.data || "{}");
  ok(bk.collections?.codes?.fake1 && bk.collections?.reviews?.fake1 && Object.keys(bk.collections?.tutorial_keys || {}).length === 8, "backup holds coding, reviews and keys");

  // ---- E. the coder can't reopen agreed coding, and removal works ----
  const c3 = open(db, coder); await sleep(150); c3.q("#tabCode").click(); await sleep(80);
  ok(!c3.btn("Reopen to edit"), "agreed coding offers no reopen");
  a.q("#tabAdmin").click(); await sleep(40);
  a.qa("tbody tr").find(tr => tr.textContent.includes("coder@uni.example.edu")).querySelector("button").click(); await sleep(60);
  ok(!db.data.has("roles/coder@uni.example.edu"), "lead removes the coder");

  fs.rmSync(out, { recursive: true, force: true });
  console.log(fails ? `${fails} FAILED` : "ALL PASSED");
  process.exit(fails ? 1 : 0);
})().catch(e => { console.error("CRASH", e); process.exit(2); });
