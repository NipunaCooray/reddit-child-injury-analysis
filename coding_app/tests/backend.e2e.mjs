// End to end against the Auth and Firestore emulators: email-link sign-in, the list check,
// and the store API the app uses, under the real rules. Run with npm run test:rules.
import { test, before, after } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { initializeTestEnvironment } from "@firebase/rules-unit-testing";
import { doc, setDoc } from "firebase/firestore";

const PROJECT = "demo-human-coding", AUTH = "http://127.0.0.1:9099", PAGE = "http://localhost:5002/";
const LEAD = "lead@example.org", CODER = "coder@uni.example.edu", OUTSIDER = "someone@example.com";
let env;
// the emulator sometimes drops a reused connection; close each one and retry once
const call = async (url, init = {}) => { const go = () => fetch(url, { ...init, headers: { ...(init.headers || {}), connection: "close" } });
  try { return await go(); } catch { await new Promise(r => setTimeout(r, 300)); return go(); } };

const page = (scenario, url, email) => JSON.parse(execFileSync("node", [fileURLToPath(new URL("./backend.page.mjs", import.meta.url)), scenario, url, email || ""],
  { cwd: fileURLToPath(new URL("..", import.meta.url)), encoding: "utf8", timeout: 30000 }).trim().split("\n").at(-1));
async function linkFor(email, send = true) {
  if (send) await call(`${AUTH}/identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key=demo-key`, { method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ requestType: "EMAIL_SIGNIN", email, continueUrl: PAGE, canHandleCodeInApp: true }) });
  const { oobCodes } = await (await call(`${AUTH}/emulator/v1/projects/${PROJECT}/oobCodes`)).json();
  const code = oobCodes.filter(c => c.email === email && c.requestType === "EMAIL_SIGNIN").at(-1).oobCode;
  return `${PAGE}?apiKey=demo-key&oobCode=${code}&mode=signIn&lang=en`;
}

before(async () => {
  env = await initializeTestEnvironment({ projectId: PROJECT, firestore: { rules: readFileSync("firestore.rules", "utf8"), host: "127.0.0.1", port: 8080 } });
  await env.clearFirestore();
  await call(`${AUTH}/emulator/v1/projects/${PROJECT}/accounts`, { method: "DELETE" });
  await env.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    await setDoc(doc(db, "roles", LEAD), { role: "admin" });
    await setDoc(doc(db, "roles", CODER), { role: "coder" });
    await setDoc(doc(db, "posts/p1"), { id: "p1", title: "t", selftext: "s", sample: "practice" });
    await setDoc(doc(db, "tutorial/t1"), { id: "t1", title: "t", selftext: "s", sample: "tutorial" });
    await setDoc(doc(db, "tutorial_keys/t1"), { label: "Near miss", key: {} });
    await setDoc(doc(db, "reviews/p1"), { post_id: "p1", state: "in_review", items: { f: { verdict: "flag", resolved: false, thread: [] } } });
  });
});
after(async () => { await env.cleanup(); });

test("the sign-in screen sends an email link and remembers the address", () => {
  const r = page("send-link", PAGE, CODER);
  assert.ok(r.sent, r.screen);
  assert.equal(r.stored, CODER);
});

test("a coder on the list signs in by link and gets exactly the coder's access", async () => {
  const r = page("coder", await linkFor(CODER, false), CODER);
  assert.equal(r.backend, true);
  assert.equal(r.role, "coder");
  assert.ok(!r.cleanedUrl.includes("oobCode"), "the one-time code is removed from the address bar");
  assert.equal(r.readPosts, 1);
  assert.equal(r.ownPeople, CODER);
  assert.equal(r.profiles, "coder");
  assert.match(r.keyBefore, /^denied/);
  assert.match(r.keyDraft, /^denied/);
  assert.equal(r.keyAfter, "Near miss", "key opens once the post is submitted");
  assert.match(r.loadPosts, /^denied/);
  assert.equal(r.code, "ok");
  assert.equal(r.reply, "ok");
  assert.equal(r.reviewAfterReply.verdict, "flag", "a thread reply keeps the reviewer's flag");
  assert.equal(r.reviewAfterReply.thread[0].text, "coder reply");
  assert.match(r.listRoles, /^denied/);
});

test("someone not on the list is signed in but sees nothing", async () => {
  const r = page("outsider", await linkFor(OUTSIDER), OUTSIDER);
  assert.equal(r.backend, false);
  assert.match(r.screen, /not on the list/);
});

test("the study lead manages people and loads files", async () => {
  const r = page("lead", await linkFor(LEAD), LEAD);
  assert.equal(r.role, "admin");
  assert.equal(r.listRoles, 2);
  assert.equal(r.loadPosts, "ok");
  assert.equal(r.readKeys, 2);
  assert.equal(r.addPerson, "ok");
  assert.equal(r.removePerson, "ok");
});
