// Signs the person in, checks they are on the list, and hands the app a small store API
// (doc/collection with set, update, delete, get, onSnapshot; batch) over Cloud Firestore.
// On localhost it talks to the Firebase emulators instead of the real project.
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.1/firebase-app.js";
import {
  getAuth, connectAuthEmulator, onAuthStateChanged, sendSignInLinkToEmail, isSignInWithEmailLink,
  signInWithEmailLink, GoogleAuthProvider, signInWithPopup, signOut,
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-auth.js";
import {
  initializeFirestore, connectFirestoreEmulator, doc, collection, getDoc, getDocs, setDoc, deleteDoc, onSnapshot, writeBatch,
} from "https://www.gstatic.com/firebasejs/10.14.1/firebase-firestore.js";

const LOCAL = ["localhost", "127.0.0.1"].includes(location.hostname);
const EMAIL_KEY = "hc-signin-email";
const main = document.getElementById("main");
const el = (tag, attrs = {}, ...kids) => { const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (v == null || v === false) continue;
    if (k === "class") n.className = v; else if (k === "text") n.textContent = v; else if (k.startsWith("on")) n.addEventListener(k.slice(2), v); else n.setAttribute(k, v === true ? "" : v); }
  n.append(...kids.filter(k => k != null)); return n; };
const screen = (...kids) => { main.className = "single"; main.replaceChildren(el("div", { class: "empty" }, ...kids)); };

let resolveBackend;
window.HC_BACKEND = new Promise(r => { resolveBackend = r; });

async function config() {
  if (LOCAL) return { apiKey: "demo-key", authDomain: "demo-human-coding.firebaseapp.com", projectId: "demo-human-coding" };
  const res = await fetch("/__/firebase/init.json"); // served by Firebase Hosting for this project
  if (!res.ok) throw new Error("no config");
  return res.json();
}

function makeStore(fs) {
  const wrapDoc = (s) => ({ id: s.id, exists: s.exists(), data: () => s.data() });
  const wrapQuery = (s) => ({ docs: s.docs.map(d => ({ id: d.id, data: () => d.data() })) });
  const docApi = (path) => { const ref = doc(fs, path); return {
    set: (d) => setDoc(ref, d),
    // nested maps merge, as the app expects (a coder's thread reply must not replace the whole review)
    update: (d) => setDoc(ref, d, { merge: true }),
    delete: () => deleteDoc(ref),
    get: async () => wrapDoc(await getDoc(ref)),
    onSnapshot: (cb, err) => onSnapshot(ref, s => cb(wrapDoc(s)), err),
  }; };
  return {
    doc: docApi,
    collection: (name) => ({
      doc: (id) => docApi(`${name}/${id}`),
      get: async () => wrapQuery(await getDocs(collection(fs, name))),
      onSnapshot: (cb, err) => onSnapshot(collection(fs, name), s => cb(wrapQuery(s)), err),
    }),
    // ops: [{op: "set" | "delete", path, data}], committed in chunks under Firestore's 500-write limit
    async batch(ops) {
      for (let i = 0; i < ops.length; i += 400) {
        const b = writeBatch(fs);
        for (const o of ops.slice(i, i + 400)) o.op === "delete" ? b.delete(doc(fs, o.path)) : b.set(doc(fs, o.path), o.data);
        await b.commit();
      }
    },
  };
}

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const a = el("a", { href: url, download: filename }); document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

function signInScreen(auth, note) {
  const email = el("input", { class: "field-input", type: "email", autocomplete: "email", placeholder: "you@example.org", "aria-label": "Email" });
  const msg = el("p", { class: "msg", text: note || "" });
  const send = async () => {
    const addr = email.value.trim().toLowerCase();
    if (!addr.includes("@")) { msg.className = "msg bad"; msg.textContent = "Enter your email address."; return; }
    try {
      await sendSignInLinkToEmail(auth, addr, { url: location.origin + location.pathname, handleCodeInApp: true });
      localStorage.setItem(EMAIL_KEY, addr);
      msg.className = "msg ok"; msg.textContent = `Sign-in link sent to ${addr}. Open it on this device. If it hasn't arrived in a few minutes, check your junk folder.`;
    } catch (e) { msg.className = "msg bad"; msg.textContent = `Couldn't send the link (${e?.code || "error"}).`; }
  };
  email.addEventListener("keydown", e => { if (e.key === "Enter") send(); });
  screen(el("h2", { text: "Sign in" }),
    el("p", { class: "muted", text: "This page is for the people on the study's coding team. Use the email address the study lead added." }),
    el("div", { class: "signin" }, email, el("button", { class: "btn primary", onclick: send, text: "Email me a sign-in link" }),
      el("button", { class: "btn", onclick: () => signInWithPopup(auth, new GoogleAuthProvider()).catch(e => { msg.className = "msg bad"; msg.textContent = `Google sign-in didn't complete (${e?.code || "error"}).`; }), text: "Sign in with Google" }),
      msg));
}

async function finishEmailLink(auth) {
  if (!isSignInWithEmailLink(auth, location.href)) return;
  let addr = localStorage.getItem(EMAIL_KEY);
  if (!addr) addr = window.prompt("Confirm the email address this link was sent to");
  if (!addr) return;
  try { await signInWithEmailLink(auth, addr.trim().toLowerCase(), location.href); localStorage.removeItem(EMAIL_KEY); }
  catch (e) { signInScreen(auth, `That sign-in link didn't work (${e?.code || "error"}). Links expire and work once; ask for a new one.`); }
  history.replaceState(null, "", location.pathname);
}

async function start() {
  let app;
  try { app = initializeApp(await config()); }
  catch { screen(el("h2", { text: "This page isn't set up yet" }), el("p", { class: "muted", text: "The Firebase configuration couldn't be loaded." })); return; }
  const auth = getAuth(app);
  const fs = initializeFirestore(app, { ignoreUndefinedProperties: true });
  if (LOCAL) { connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true }); connectFirestoreEmulator(fs, "127.0.0.1", 8080); }
  await finishEmailLink(auth);

  let started = false;
  onAuthStateChanged(auth, async (user) => {
    if (started) { if (!user) location.reload(); return; } // signed out mid-session: start again
    if (!user) return signInScreen(auth);
    const email = (user.email || "").toLowerCase();
    const out = el("button", { class: "btn", onclick: () => signOut(auth), text: "Sign out" });
    let role = null;
    try { const r = await getDoc(doc(fs, "roles", email)); role = r.exists() ? r.data().role : null; } catch { role = null; }
    if (!role || !user.emailVerified) return screen(el("h2", { text: "You're signed in, but not on the list" }),
      el("p", { class: "muted", text: `Signed in as ${email || "an account with no email"}. Ask the study lead to add this exact address, then reload. If they added a different address, sign out and use that one.` }), out);
    const name = user.displayName || email.split("@")[0];
    try { await setDoc(doc(fs, "people", user.uid), { name, email, updated_at: new Date().toISOString() }); } catch { /* names are a nicety */ }
    const store = makeStore(fs), names = new Map();
    const backend = {
      db: store, role, email, download, signOut: () => signOut(auth),
      user: {
        me: async () => ({ id: user.uid, name }),
        // {id: {name}} for the ids given, read from people/
        profiles: async (ids) => {
          const out = {};
          for (const id of [].concat(ids)) {
            if (!names.has(id)) { try { const p = await getDoc(doc(fs, "people", id)); names.set(id, p.exists() ? p.data().name : ""); } catch { names.set(id, ""); } }
            out[id] = { id, name: names.get(id) };
          }
          return out;
        },
      },
    };
    started = true;
    resolveBackend(backend);
    window.dispatchEvent(new CustomEvent("hc-backend", { detail: backend }));
  });
}
start();
