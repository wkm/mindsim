/**
 * Quick Switcher (Ctrl+K) — fuzzy search across bots and components.
 *
 * Fetches /api/bots and /api/components on first open, then filters
 * by name and category. Arrow keys + Enter to navigate, Escape to close.
 */

let items = null;  // [{name, category, url}]
let activeIdx = 0;

const overlay = document.getElementById('quick-switcher-overlay');
const input = document.getElementById('qs-input');
const results = document.getElementById('qs-results');

async function ensureItems() {
  if (items) return;
  const [botsResp, compsResp] = await Promise.all([
    fetch('/api/bots'),
    fetch('/api/components'),
  ]);
  const bots = await botsResp.json();
  const comps = await compsResp.json();

  items = [];
  for (const b of bots) {
    items.push({ name: b.name, category: 'bot', url: `?bot=${b.name}` });
  }
  for (const c of comps) {
    items.push({ name: c.name, category: c.category, url: `?component=${c.name}` });
  }
}

function fuzzyMatch(query, item) {
  const q = query.toLowerCase();
  const name = item.name.toLowerCase();
  const cat = item.category.toLowerCase();
  // Match if query appears in name or category
  return name.includes(q) || cat.includes(q);
}

function render(query) {
  const filtered = query
    ? items.filter(item => fuzzyMatch(query, item))
    : items;

  if (!filtered.length) {
    results.innerHTML = '<div class="qs-empty">No results</div>';
    activeIdx = -1;
    return;
  }

  activeIdx = Math.min(activeIdx, filtered.length - 1);
  if (activeIdx < 0) activeIdx = 0;

  results.innerHTML = filtered.map((item, i) => `
    <div class="qs-item${i === activeIdx ? ' qs-active' : ''}" data-url="${item.url}">
      <span class="qs-item-name">${item.name}</span>
      <span class="qs-item-category">${item.category}</span>
    </div>
  `).join('');

  // Click handler
  results.querySelectorAll('.qs-item').forEach(el => {
    el.addEventListener('click', () => navigate(el.dataset.url));
  });

  // Scroll active into view
  const active = results.querySelector('.qs-active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function navigate(url) {
  window.location.search = url;
}

function open() {
  overlay.style.display = 'flex';
  input.value = '';
  activeIdx = 0;
  ensureItems().then(() => render(''));
  // Focus after display so it works
  requestAnimationFrame(() => input.focus());
}

function close() {
  overlay.style.display = 'none';
}

// Input filtering
input.addEventListener('input', () => {
  activeIdx = 0;
  if (items) render(input.value);
});

// Keyboard navigation
input.addEventListener('keydown', (e) => {
  const filtered = items
    ? (input.value ? items.filter(item => fuzzyMatch(input.value, item)) : items)
    : [];

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    activeIdx = Math.min(activeIdx + 1, filtered.length - 1);
    render(input.value);
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    activeIdx = Math.max(activeIdx - 1, 0);
    render(input.value);
  } else if (e.key === 'Enter') {
    e.preventDefault();
    if (activeIdx >= 0 && activeIdx < filtered.length) {
      navigate(filtered[activeIdx].url);
    }
  } else if (e.key === 'Escape') {
    e.preventDefault();
    close();
  }
});

// Click outside to close
overlay.addEventListener('click', (e) => {
  if (e.target === overlay) close();
});

// Global Ctrl+K / Cmd+K
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
    e.preventDefault();
    if (overlay.style.display !== 'flex') {
      open();
    } else {
      close();
    }
  }
});
