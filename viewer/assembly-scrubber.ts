/**
 * Assembly Step Scrubber — slider for stepping through the assembly sequence.
 *
 * At each step, only geometry installed so far is visible. Provides a range
 * slider, prev/next buttons, a "Show All" shortcut, and step description text.
 */

export interface AssemblyOpData {
  step: number;
  action: string;
  body: string;
  description: string;
  tool: string | null;
}

export type StepChangeHandler = (step: number) => void;

export class AssemblyScrubber {
  private container: HTMLElement;
  private onStepChange: StepChangeHandler;
  private ops: AssemblyOpData[] = [];
  private step = 0;
  private el: HTMLElement | null = null;
  private slider: HTMLInputElement | null = null;
  private counterEl: HTMLElement | null = null;
  private descEl: HTMLElement | null = null;

  constructor(container: HTMLElement, onStepChange: StepChangeHandler) {
    this.container = container;
    this.onStepChange = onStepChange;
  }

  setOps(ops: AssemblyOpData[]): void {
    this.ops = ops;
    this.step = ops.length > 0 ? ops.length - 1 : 0;
    this.render();
    this.updateDisplay();
  }

  setStep(step: number): void {
    if (this.ops.length === 0) return;
    this.step = Math.max(0, Math.min(step, this.ops.length - 1));
    this.updateDisplay();
    this.onStepChange(this.step);
  }

  currentStep(): number {
    return this.step;
  }

  dispose(): void {
    if (this.el) {
      this.el.remove();
      this.el = null;
    }
    this.slider = null;
    this.counterEl = null;
    this.descEl = null;
  }

  // ---------------------------------------------------------------------------
  // Internal
  // ---------------------------------------------------------------------------

  private render(): void {
    // Remove previous DOM if any
    if (this.el) this.el.remove();

    const maxStep = Math.max(0, this.ops.length - 1);

    const bar = document.createElement('div');
    bar.className = 'assembly-scrubber';
    bar.style.cssText = [
      'position: absolute',
      'bottom: 0',
      'left: 0',
      'right: 0',
      'z-index: 90',
      'display: flex',
      'align-items: center',
      'gap: 6px',
      'padding: 8px 16px',
      'background: rgba(255,255,255,0.92)',
      'backdrop-filter: blur(8px)',
      'border-top: 1px solid var(--border)',
      'font-family: var(--font)',
      'font-size: 12px',
    ].join(';');

    // Prev button
    const prevBtn = document.createElement('button');
    prevBtn.className = 'btn-ghost btn-sm';
    prevBtn.textContent = '\u25C0';
    prevBtn.title = 'Previous step';
    prevBtn.addEventListener('click', () => this.setStep(this.step - 1));
    bar.appendChild(prevBtn);

    // Slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = '0';
    slider.max = String(maxStep);
    slider.step = '1';
    slider.value = String(this.step);
    slider.style.cssText = 'flex: 1; min-width: 100px;';
    slider.addEventListener('input', () => {
      this.step = Number.parseInt(slider.value, 10);
      this.updateDisplay();
      this.onStepChange(this.step);
    });
    bar.appendChild(slider);
    this.slider = slider;

    // Next button
    const nextBtn = document.createElement('button');
    nextBtn.className = 'btn-ghost btn-sm';
    nextBtn.textContent = '\u25B6';
    nextBtn.title = 'Next step';
    nextBtn.addEventListener('click', () => this.setStep(this.step + 1));
    bar.appendChild(nextBtn);

    // Counter
    const counter = document.createElement('span');
    counter.style.cssText = [
      'white-space: nowrap',
      'color: var(--muted-fg)',
      'font-family: var(--font-mono)',
      'min-width: 80px',
      'text-align: center',
    ].join(';');
    bar.appendChild(counter);
    this.counterEl = counter;

    // Show All button
    const showAllBtn = document.createElement('button');
    showAllBtn.className = 'btn-ghost btn-sm';
    showAllBtn.textContent = 'Show All';
    showAllBtn.title = 'Show all assembly steps';
    showAllBtn.addEventListener('click', () => this.setStep(maxStep));
    bar.appendChild(showAllBtn);

    // Description (second row, spans full width)
    const desc = document.createElement('div');
    desc.style.cssText = [
      'position: absolute',
      'bottom: 100%',
      'left: 0',
      'right: 0',
      'padding: 4px 16px',
      'background: rgba(255,255,255,0.88)',
      'backdrop-filter: blur(8px)',
      'border-top: 1px solid var(--border)',
      'font-size: 12px',
      'color: var(--dark5)',
      'white-space: nowrap',
      'overflow: hidden',
      'text-overflow: ellipsis',
    ].join(';');
    bar.appendChild(desc);
    this.descEl = desc;

    this.container.appendChild(bar);
    this.el = bar;
  }

  private updateDisplay(): void {
    if (!this.slider || !this.counterEl || !this.descEl) return;

    const total = this.ops.length;
    this.slider.value = String(this.step);
    this.slider.max = String(Math.max(0, total - 1));
    this.counterEl.textContent = `Step ${this.step + 1} of ${total}`;

    const op = this.ops[this.step];
    if (op) {
      const toolStr = op.tool ? ` [${op.tool}]` : '';
      this.descEl.textContent = `${op.action}: ${op.body} \u2014 ${op.description}${toolStr}`;
    } else {
      this.descEl.textContent = '';
    }
  }
}
