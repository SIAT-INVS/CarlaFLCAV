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
