"use strict";
const nav = document.querySelector('.nav');
const toggle = document.querySelector('.menu-toggle');
if (nav && toggle) {
  const close = () => toggle.setAttribute('aria-expanded', 'false');
  toggle.hidden = false;
  nav.classList.add('menu-ready');
  toggle.addEventListener('click', () => toggle.setAttribute('aria-expanded', String(toggle.getAttribute('aria-expanded') !== 'true')));
  nav.addEventListener('click', e => { if (e.target.closest('a')) close(); });
  document.addEventListener('click', e => { if (!nav.contains(e.target)) close(); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') { close(); toggle.focus(); }
  });
  nav.addEventListener('focusout', e => { if (!nav.contains(e.relatedTarget)) close(); });
  window.matchMedia('(max-width: 950px)').addEventListener('change', close);
}

// Fetch the published repository snapshot, not two rate-limited API calls per visitor.
// The bundled snapshot remains usable offline or while GitHub is unavailable.
async function refreshStars() {
  const names = ['SIAT-INVS/CarlaFLCAV', 'KevinLADLee/carla_dataset_tools'];
  const sources = ['https://raw.githubusercontent.com/SIAT-INVS/CarlaFLCAV/main/docs/assets/stars.json', 'assets/stars.json'];
  for (const source of sources) {
    try {
      const response = await fetch(source, {cache: 'no-store', signal: AbortSignal.timeout(5000)});
      if (!response.ok) continue;
      const data = await response.json();
      const values = names.map(name => data.repositories?.[name]);
      if (!values.every(n => Number.isInteger(n) && n >= 0) || data.total !== values[0] + values[1]) continue;
      document.querySelector('[data-stars="total"]').textContent = data.total.toLocaleString('en-US');
      return;
    } catch { /* Keep the last known counts on network errors. */ }
  }
}
refreshStars();

document.querySelectorAll('[data-copy]').forEach(button => {
  button.addEventListener('click', async () => {
    const code = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(code.textContent.trim());
      document.querySelector('#copy-status').textContent = 'BibTeX copied to clipboard.';
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      document.querySelector('#copy-status').textContent = 'Citation selected. Use your browser’s Copy command.';
    }
  });
});
