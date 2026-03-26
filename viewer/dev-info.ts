/**
 * Dev info banner — shows git branch and worktree when not on master/main.
 *
 * Fetches /api/dev-info on load and, for non-default branches, displays a
 * persistent gold banner below the top bar and updates the page title.
 */

async function initDevInfo() {
  try {
    const res = await fetch('/api/dev-info');
    if (!res.ok) return;
    const info: { branch: string; worktree: string; is_default: boolean } = await res.json();
    if (info.is_default) return;

    // Extract worktree leaf name (e.g. "mindsim-my-feature" from full path)
    const worktreeName = info.worktree.split('/').pop() ?? info.worktree;
    const isWorktree = worktreeName !== 'mindsim';

    const banner = document.getElementById('branch-banner');
    if (!banner) return;

    const parts: string[] = [`\u2022 ${info.branch}`];
    if (isWorktree) {
      parts.push(`[${worktreeName}]`);
    }
    banner.textContent = parts.join('  ');
    banner.style.display = 'block';
    document.body.classList.add('has-branch-banner');

    // Update page title
    document.title = `${info.branch} — MindSim`;
  } catch {
    // Dev info is best-effort; don't break the viewer
  }
}

initDevInfo();
