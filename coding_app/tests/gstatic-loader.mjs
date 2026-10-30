// Node module hook: serve backend.js's CDN imports from the installed firebase package (same version)
export async function resolve(specifier, context, next) {
  const m = specifier.match(/^https:\/\/www\.gstatic\.com\/firebasejs\/[\d.]+\/firebase-(app|auth|firestore)\.js$/);
  return next(m ? `firebase/${m[1]}` : specifier, context);
}
