// Security rules for the coding app, run against the Firestore emulator (npm run test:rules).
import { test, before, after, beforeEach } from "node:test";
import { readFileSync } from "node:fs";
import { initializeTestEnvironment, assertSucceeds, assertFails } from "@firebase/rules-unit-testing";
import { doc, getDoc, setDoc, deleteDoc, collection, getDocs } from "firebase/firestore";

let env;
const PEOPLE = {
  lead: { uid: "uLead", email: "lead@example.org", role: "admin" },
  coder: { uid: "uCoder", email: "coder@uni.example.edu", role: "coder" },
  reviewer: { uid: "uRev", email: "reviewer@example.org", role: "reviewer" },
  outsider: { uid: "uOut", email: "someone@example.com", role: null },
};
const as = (p, verified = true) => env.authenticatedContext(p.uid, { email: p.email, email_verified: verified }).firestore();
const code = (uid, post = "p1", extra = {}) => ({ post_id: post, coder: uid, state: "draft", fields: {}, sample: "practice", ...extra });

before(async () => {
  env = await initializeTestEnvironment({ projectId: "demo-human-coding", firestore: { rules: readFileSync("firestore.rules", "utf8"), host: "127.0.0.1", port: 8080 } });
});
after(async () => { await env.cleanup(); });
beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async (ctx) => {
    const db = ctx.firestore();
    for (const p of Object.values(PEOPLE)) if (p.role) await setDoc(doc(db, "roles", p.email), { role: p.role });
    await setDoc(doc(db, "posts/p1"), { id: "p1", title: "t", selftext: "s", sample: "practice" });
    await setDoc(doc(db, "config/codebook"), { schema_version: "1.5.0" });
    await setDoc(doc(db, "tutorial/t1"), { id: "t1", title: "t", selftext: "s" });
    await setDoc(doc(db, "tutorial_keys/t1"), { label: "Near miss", key: { f: { value: "x" } } });
  });
});

test("outsiders, signed-out visitors and unverified emails read nothing", async () => {
  await assertFails(getDoc(doc(as(PEOPLE.outsider), "posts/p1")));
  await assertFails(getDoc(doc(env.unauthenticatedContext().firestore(), "posts/p1")));
  await assertFails(getDoc(doc(as(PEOPLE.coder, false), "posts/p1")));
  await assertFails(setDoc(doc(as(PEOPLE.outsider), "codes/p1"), code("uOut")));
});

test("members read posts and the codebook but only the lead loads them", async () => {
  await assertSucceeds(getDocs(collection(as(PEOPLE.coder), "posts")));
  await assertSucceeds(getDoc(doc(as(PEOPLE.reviewer), "config/codebook")));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "posts/p2"), { title: "x" }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "config/codebook"), { schema_version: "9" }));
  await assertSucceeds(setDoc(doc(as(PEOPLE.lead), "posts/p2"), { title: "x" }));
});

test("only the lead manages the list; members see only their own entry", async () => {
  await assertSucceeds(getDoc(doc(as(PEOPLE.coder), "roles", PEOPLE.coder.email)));
  await assertFails(getDoc(doc(as(PEOPLE.coder), "roles", PEOPLE.lead.email)));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "roles", PEOPLE.coder.email), { role: "admin" }));
  await assertFails(setDoc(doc(as(PEOPLE.outsider), "roles", PEOPLE.outsider.email), { role: "coder" }));
  await assertSucceeds(setDoc(doc(as(PEOPLE.lead), "roles", "new@example.org"), { role: "coder" }));
  await assertFails(setDoc(doc(as(PEOPLE.lead), "roles", "new@example.org"), { role: "owner" }));
  await assertSucceeds(getDocs(collection(as(PEOPLE.lead), "roles")));
});

test("the coder writes their own coding; nobody else can", async () => {
  await assertSucceeds(setDoc(doc(as(PEOPLE.coder), "codes/p1"), code("uCoder")));
  await assertFails(setDoc(doc(as(PEOPLE.reviewer), "codes/p1"), code("uRev")));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "codes/p1"), code("uRev")));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "codes/p2"), code("uCoder", "p1")));
  await assertFails(deleteDoc(doc(as(PEOPLE.coder), "codes/p1")));
});

test("coding locks once the review is agreed", async () => {
  await assertSucceeds(setDoc(doc(as(PEOPLE.coder), "codes/p1"), code("uCoder", "p1", { state: "submitted" })));
  await assertSucceeds(setDoc(doc(as(PEOPLE.reviewer), "reviews/p1"), { post_id: "p1", state: "agreed", items: {} }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "codes/p1"), code("uCoder", "p1", { state: "draft" })));
});

test("the reviewer reviews; the coder may only reply in the threads", async () => {
  await assertFails(setDoc(doc(as(PEOPLE.coder), "reviews/p1"), { post_id: "p1", state: "in_review", items: {} }));
  await assertSucceeds(setDoc(doc(as(PEOPLE.reviewer), "reviews/p1"), { post_id: "p1", state: "in_review", items: { f: { verdict: "flag", thread: [] } } }));
  await assertSucceeds(setDoc(doc(as(PEOPLE.coder), "reviews/p1"), { items: { f: { thread: [{ text: "reply" }] } }, updated_at: "now" }, { merge: true }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "reviews/p1"), { state: "agreed" }, { merge: true }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "reviews/p1"), { reviewer: "uCoder" }, { merge: true }));
});

test("tutorial attempts are per person", async () => {
  await assertSucceeds(setDoc(doc(as(PEOPLE.coder), "tutorial_codes/t1--uCoder"), code("uCoder", "t1", { sample: "tutorial" })));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "tutorial_codes/t1--uRev"), code("uRev", "t1", { sample: "tutorial" })));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "tutorial_codes/t1--uRev"), code("uCoder", "t1", { sample: "tutorial" })));
  await assertFails(setDoc(doc(as(PEOPLE.outsider), "tutorial_codes/t1--uOut"), code("uOut", "t1", { sample: "tutorial" })));
});

test("the tutorial key opens only after that person submits the post", async () => {
  const coder = as(PEOPLE.coder);
  await assertFails(getDoc(doc(coder, "tutorial_keys/t1")));
  await assertSucceeds(setDoc(doc(coder, "tutorial_codes/t1--uCoder"), code("uCoder", "t1", { sample: "tutorial" })));
  await assertFails(getDoc(doc(coder, "tutorial_keys/t1")));
  await assertSucceeds(setDoc(doc(coder, "tutorial_codes/t1--uCoder"), code("uCoder", "t1", { sample: "tutorial", state: "submitted" })));
  await assertSucceeds(getDoc(doc(coder, "tutorial_keys/t1")));
  // someone else's submission doesn't open it for the reviewer
  await assertFails(getDoc(doc(as(PEOPLE.reviewer), "tutorial_keys/t1")));
  await assertFails(getDocs(collection(coder, "tutorial_keys")));
  await assertSucceeds(getDocs(collection(as(PEOPLE.lead), "tutorial_keys")));
  await assertFails(setDoc(doc(coder, "tutorial_keys/t1"), { key: {} }));
});

test("people write only their own display name", async () => {
  await assertSucceeds(setDoc(doc(as(PEOPLE.coder), "people/uCoder"), { name: "Coder", email: PEOPLE.coder.email, updated_at: "now" }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "people/uRev"), { name: "x", email: "x", updated_at: "now" }));
  await assertFails(setDoc(doc(as(PEOPLE.coder), "people/uCoder"), { name: "x", role: "admin" }));
  await assertFails(setDoc(doc(as(PEOPLE.outsider), "people/uOut"), { name: "x", email: "x", updated_at: "now" }));
  await assertSucceeds(getDoc(doc(as(PEOPLE.reviewer), "people/uCoder")));
});
