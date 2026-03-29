/**
 * Shared syntax highlighting for assembly repr output.
 *
 * Used by assembly-viewer.ts and assembly-scrubber.ts for consistent
 * highlighting of Python repr strings (AssemblyOp, etc.).
 */

/** Escape HTML special characters to prevent injection. */
export function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

/** Syntax-highlight a Python repr string with inline styles. */
export function highlightRepr(repr: string): string {
  let s = escapeHtml(repr);

  // Token replacement to avoid double-matching
  const tokens: { key: string; html: string }[] = [];
  let tokenId = 0;
  const placeholder = (cls: string, text: string) => {
    const key = `\x00T${tokenId++}\x00`;
    tokens.push({ key, html: `<span style="${cls}">${text}</span>` });
    return key;
  };

  // Strings (single and double quoted)
  s = s.replace(/'[^']*'/g, (m) => placeholder('color:#D4A72C', m));
  s = s.replace(/"[^"]*"/g, (m) => placeholder('color:#D4A72C', m));

  // Class names / type names (PascalCase identifiers)
  s = s.replace(/\b(AssemblyOp|AssemblyAction|ComponentRef|FastenerRef|WireRef|JointId|BodyId|ToolKind)\b/g, (m) =>
    placeholder('color:#2B95D6;font-weight:600', m),
  );

  // Enum values (UPPER_CASE after a dot)
  s = s.replace(/\.([A-Z][A-Z0-9_]+)\b/g, (_m, val) => `.${placeholder('color:#0F9960;font-weight:600', val)}`);

  // Numbers (integers and floats, including negatives)
  s = s.replace(/\b(-?\d+\.?\d*)\b/g, (m) => placeholder('color:#C87619', m));

  // None, True, False
  s = s.replace(/\b(None|True|False)\b/g, (m) => placeholder('color:#9179F2', m));

  // Replace placeholders with actual HTML
  for (const { key, html } of tokens) {
    s = s.replace(key, html);
  }

  return s;
}
