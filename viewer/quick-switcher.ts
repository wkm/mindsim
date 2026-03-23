/**
 * Quick Switcher (Ctrl+K) — fuzzy search across bots and components.
 *
 * Fetches /api/bots and /api/components on first open, then filters
 * by name and category. Arrow keys + Enter to navigate, Escape to close.
 */

interface QSItem {
  name: string;
  category: string;
  url: string;
}

let items: QSItem[] | null = null;
let filtered: QSItem[] = [];
let activeIdx = 0;

const overlay = document.getElementById('quick-switcher-overlay')!;
const input = document.getElementById('qs-input') as HTMLInputElement;
const results = document.getElementById('qs-results')!;

async function ensureItems() {
  if (items) return;
  const [botsResp, compsResp] = await Promise.all([fetch('/api/bots'), fetch('/api/components')]);
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

function fuzzyMatch(query: string, item: QSItem) {
  const q = query.toLowerCase();
  return item.name.toLowerCase().includes(q) || item.category.toLowerCase().includes(q);
}

function refilter(query: string) {
  filtered = query ? items!.filter((item) => fuzzyMatch(query, item)) : [...items!];
  activeIdx = Math.max(0, Math.min(activeIdx, filtered.length - 1));
  if (!filtered.length) activeIdx = -1;
}

function render() {
  if (!filtered.length) {
    results.innerHTML = '<li class="dropdown-item disabled"><span class="text">No results</span></li>';
    return;
  }

  results.innerHTML = filtered
    .map(
      (item, i) => `
    <li><button class="dropdown-item${i === activeIdx ? ' active' : ''}" data-url="${item.url}">
      <span class="text">${item.name}</span>
      <span class="dropdown-kbd">${item.category}</span>
    </button></li>
  `,
    )
    .join('');

  const active = results.querySelector('.active');
  if (active) active.scrollIntoView({ block: 'nearest' });
}

function navigate(url: string) {
  window.location.href = url;
}

function open() {
  overlay.style.display = 'flex';
  input.value = '';
  activeIdx = 0;
  ensureItems().then(() => {
    refilter('');
    render();
  });
  requestAnimationFrame(() => input.focus());
}

function close() {
  overlay.style.display = 'none';
}

// Input filtering
input.addEventListener('input', () => {
  activeIdx = 0;
  if (items) {
    refilter(input.value);
    render();
  }
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
  const item = (e.target as HTMLElement).closest('.dropdown-item') as HTMLElement | null;
  if (item) navigate(item.dataset.url!);
});

// Click outside to close
overlay.addEventListener('click', (e) => {
  if (e.target === overlay) close();
});

// Bot/component name in navbar opens switcher
document.getElementById('bot-name')!.addEventListener('click', () => open());

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
