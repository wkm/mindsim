/**
 * Quick Switcher (Ctrl+K) — fuzzy search across bots and components.
 *
 * Fetches /api/bots and /api/components on first open, then filters
 * by name and category. Arrow keys + Enter to navigate, Escape to close.
 */

let items = null;  // [{name, category, url}]
let filtered = []; // current filtered view
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
  return item.name.toLowerCase().includes(q) || item.category.toLowerCase().includes(q);
}

function refilter(query) {
  filtered = query ? items.filter(item => fuzzyMatch(query, item)) : [...items];
  activeIdx = Math.max(0, Math.min(activeIdx, filtered.length - 1));
  if (!filtered.length) activeIdx = -1;
}

function render() {
  if (!filtered.length) {
    results.innerHTML = '<li class="bp5-menu-item bp5-disabled"><span class="bp5-text">No results</span></li>';
    return;
  }

  results.innerHTML = filtered.map((item, i) => `
    <li><button class="bp5-menu-item${i === activeIdx ? ' bp5-active' : ''}" data-url="${item.url}">
      <span class="bp5-text">${item.name}</span>
      <span class="bp5-menu-item-label">${item.category}</span>
    </button></li>
  `).join('');

  const active = results.querySelector('.bp5-active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function navigate(url) {
  window.location.href = url;
}

function open() {
  overlay.style.display = 'flex';
  input.value = '';
  activeIdx = 0;
  ensureItems().then(() => { refilter(''); render(); });
  requestAnimationFrame(() => input.focus());
}

function close() {
  overlay.style.display = 'none';
}

// Input filtering
input.addEventListener('input', () => {
  activeIdx = 0;
  if (items) { refilter(input.value); render(); }
});

// Keyboard navigation
input.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    activeIdx = Math.min(activeIdx + 1, filtered.length - 1);
    render();
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    activeIdx = Math.max(activeIdx - 1, 0);
    render();
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

// Delegated click handler on results container
results.addEventListener('click', (e) => {
  const item = e.target.closest('.bp5-menu-item');
  if (item) navigate(item.dataset.url);
});

// Click outside to close
overlay.addEventListener('click', (e) => {
  if (e.target === overlay) close();
});

// Bot/component name in navbar opens switcher
document.getElementById('bot-name').addEventListener('click', () => open());

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
