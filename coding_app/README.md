# Human coding app

The web app the human coder and reviewer use for Phase 2: https://human-coding.web.app

It is a single page on Firebase Hosting, with the data in Cloud Firestore (Firebase project `human-coding`, region `australia-southeast1`). The design it implements, one blind coder plus a reviewer coding to consensus, is in `src/extract/HAND_REVIEW.md` and the study guide.

## Who can do what

Nobody can read anything until the study lead adds their email in the Admin tab. People sign in with a link emailed to them, or with Google if the address is a Google account. The database's access rules (`firestore.rules`) enforce the roles, not just the page:

| Role | Can |
|---|---|
| `coder` | Do the tutorial, code posts, and reply to the reviewer's notes. Can't change the coding once its review is agreed. |
| `reviewer` | Do the tutorial and review coded posts. Can't code. |
| `admin` | Everything, including the Admin tab: people, loading files, backups. |

Everyone on the list can read the posts, the codebook and everyone's coding and reviews. A tutorial answer key opens only once that person has submitted that tutorial post.

## Files

| File | What it is |
|---|---|
| `public/index.html` | The app: Tutorial, Code, Review, Progress and Admin tabs. |
| `public/backend.js` | Sign-in, the list check, and the small store API the app uses. On `localhost` it talks to the emulators. |
| `firestore.rules` | Who can read and write what. |
| `tests/rules.test.mjs` | The access rules, role by role, on the Firestore emulator. |
| `tests/backend.e2e.mjs` | Email-link sign-in, the list check and the store API, on the Auth and Firestore emulators. |
| `tests/app.smoke.cjs` | The app in a simulated browser: loading files, the tutorial, coding, review with a coder's reply, export and backup. |

Post text never goes in git. The study lead exports it locally with `src/extract/export_coding_app.py` (into `reports/`, git-ignored) and loads it in the Admin tab.

## One-time setup (already done for `human-coding` except where marked)

1. Create the Firebase project and a Firestore database in `australia-southeast1`.
2. Register a web app (`firebase apps:create web "Coding app"`). The page reads its settings from Hosting's `/__/firebase/init.json`.
3. **Console, Authentication, Sign-in method:** turn on **Email/Password** with **Email link (passwordless sign-in)**, and **Google**. *(Study lead)*
4. **Console, Firestore, Data:** add the first admin by hand, because the rules let only an admin add people. Create collection `roles`, document ID = your email in lower case, field `role` (string) = `admin`. *(Study lead)*
5. Deploy: `npm run deploy` (needs `firebase login`).

## Loading a sample

```bash
python src/extract/export_coding_app.py --posts data/interim/practice13_stratified.jsonl --sample practice
```

Then, in the Admin tab, choose the files under `reports/coding_app/`: `codebook.json`, the files in `tutorial/`, and the sample's files in `posts/<sample>/` (or choose the whole `reports/coding_app` folder). The app shows what it will load and warns before replacing anything. Coding and reviews are never touched by a load.

To remove a post, delete it in the Firebase console (Firestore, `posts`).

## Backups

Admin tab, **Download a full backup**: one JSON file with every collection (people, posts, codebook, tutorial, coding, reviews). Do it after each coding session and keep it with the study's data, not in git. Progress, **Download agreed coding**, gives the agreed coding for one sample, which is what the statistics use.

## Tests

```bash
cd coding_app && npm install
npm test            # app smoke test, then the rules and end-to-end tests on the emulators
```

The emulator tests need Java (21+ recommended; Firebase's tools will require it from version 15) and run under the demo project `demo-human-coding`, so they never touch the real one.
