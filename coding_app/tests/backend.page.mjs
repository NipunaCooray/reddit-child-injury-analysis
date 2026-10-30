// Runs public/backend.js once, as if in a browser at the given URL, against the emulators.
// argv: <scenario> <url> [email]. Prints one JSON line with what happened, then exits.
// One process per page load, because the Firebase SDK keeps one app per process.
import { register } from "node:module";
import { JSDOM } from "jsdom";
register("./gstatic-loader.mjs", import.meta.url);

const [scenario, url, email] = process.argv.slice(2);
const dom = new JSDOM(`<!doctype html><header></header><main id="main"></main>`, { url, pretendToBeVisual: true });
const w = dom.window;
for (const k of ["window", "document", "location", "history", "localStorage", "CustomEvent", "HTMLElement"]) globalThis[k] = k === "window" ? w : w[k];
if (email) w.localStorage.setItem("hc-signin-email", email);
const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const text = () => w.document.getElementById("main").textContent;
const report = (o) => { console.log(JSON.stringify(o)); process.exit(0); };
const until = async (cond, ms = 8000) => { const t = Date.now(); while (Date.now() - t < ms) { if (await cond()) return true; await sleep(100); } return false; };

await import("../public/backend.js");
const backendOrNull = Promise.race([w.HC_BACKEND, sleep(8000).then(() => null)]);

if (scenario === "send-link") {
  await until(() => text().includes("Sign in"));
  const input = w.document.querySelector('input[type="email"]');
  input.value = email;
  [...w.document.querySelectorAll("button")].find(b => b.textContent.startsWith("Email me")).click();
  const sent = await until(() => text().includes("Sign-in link sent"));
  report({ screen: text().slice(0, 200), sent, stored: w.localStorage.getItem("hc-signin-email") });
}

const be = await backendOrNull;
if (!be) report({ backend: false, screen: text().slice(0, 300), cleanedUrl: w.location.href });

const out = { backend: true, role: be.role, email: be.email, cleanedUrl: w.location.href };
const me = await be.user.me(); out.me = me;
const tryIt = async (name, fn) => { try { out[name] = (await fn()) ?? "ok"; } catch (e) { out[name] = "denied:" + (e?.code || e?.message); } };
const db = be.db;

if (scenario === "coder") {
  await tryIt("readPosts", async () => (await db.collection("posts").get()).docs.length);
  await tryIt("ownPeople", async () => (await db.doc("people/" + me.id).get()).data().email);
  await tryIt("profiles", async () => (await be.user.profiles([me.id]))[me.id].name);
  await tryIt("keyBefore", async () => (await db.doc("tutorial_keys/t1").get()).data().label);
  await tryIt("draft", () => db.doc(`tutorial_codes/t1--${me.id}`).set({ post_id: "t1", coder: me.id, state: "draft", sample: "tutorial" }));
  await tryIt("keyDraft", async () => (await db.doc("tutorial_keys/t1").get()).data().label);
  await tryIt("submit", () => db.doc(`tutorial_codes/t1--${me.id}`).set({ post_id: "t1", coder: me.id, state: "submitted", sample: "tutorial" }));
  // the app's way: a live listener, opened after the submission
  out.keyAfter = await new Promise(res => { const off = db.doc("tutorial_keys/t1").onSnapshot(s => { off(); res(s.exists ? s.data().label : "missing"); }, e => res("denied:" + e.code)); });
  await tryIt("loadPosts", () => db.batch([{ op: "set", path: "posts/x", data: { id: "x" } }]));
  await tryIt("code", () => db.doc("codes/p1").set({ post_id: "p1", coder: me.id, state: "submitted", fields: {} }));
  await tryIt("reply", () => db.doc("reviews/p1").update({ items: { f: { thread: [{ text: "coder reply" }] } }, updated_at: "now" }));
  await tryIt("reviewAfterReply", async () => (await db.doc("reviews/p1").get()).data().items.f);
  await tryIt("listRoles", async () => (await db.collection("roles").get()).docs.length);
}
if (scenario === "lead") {
  await tryIt("listRoles", async () => (await db.collection("roles").get()).docs.length);
  await tryIt("loadPosts", () => db.batch([{ op: "set", path: "posts/x", data: { id: "x", sample: "practice" } }, { op: "set", path: "tutorial_keys/t2", data: { label: "L" } }]));
  await tryIt("readKeys", async () => (await db.collection("tutorial_keys").get()).docs.length);
  await tryIt("addPerson", () => db.doc("roles/new@example.org").set({ role: "reviewer" }));
  await tryIt("removePerson", () => db.doc("roles/new@example.org").delete());
}
report(out);
