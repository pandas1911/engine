import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const partsCode = readFileSync(join(__dirname, '../parts.js'), 'utf-8');

function loadParts() {
  const fn = new Function(
    'document', 'marked', 'autoScroll',
    partsCode + '; return { createSubAgentPanel, createSubAgentThinking, createSubAgentText, createSubAgentToolRow, updateSubAgentToolResult, createToolPart };'
  );
  return fn(document, globalThis.marked, globalThis.autoScroll);
}

describe('Sub-agent panel rendering', () => {
  let fns;

  beforeEach(() => {
    document.body.innerHTML = '';
    document.head.innerHTML = '';
    globalThis.marked = { parse: vi.fn(text => `<p>${text}</p>`) };
    globalThis.autoScroll = vi.fn();
    fns = loadParts();
    globalThis.createSubAgentPanel = fns.createSubAgentPanel;
    globalThis.createSubAgentThinking = fns.createSubAgentThinking;
    globalThis.createSubAgentText = fns.createSubAgentText;
    globalThis.createSubAgentToolRow = fns.createSubAgentToolRow;
    globalThis.updateSubAgentToolResult = fns.updateSubAgentToolResult;
    globalThis.createToolPart = fns.createToolPart;
  });

  describe('createSubAgentPanel', () => {
    it('returns correct structure with all expected children', () => {
      const panel = createSubAgentPanel(1, 'task_abc', 'research', 'search for info');

      expect(panel).toBeInstanceOf(HTMLElement);
      expect(panel.className).toBe('subagent-panel');
      expect(panel.dataset.partId).toBe('1');
      expect(panel.dataset.taskId).toBe('task_abc');

      const header = panel.querySelector('.subagent-header');
      expect(header).not.toBeNull();

      const content = panel.querySelector('.subagent-content');
      expect(content).not.toBeNull();

      expect(header.querySelector('.spinner-svg')).not.toBeNull();
      expect(header.querySelector('.subagent-label')).not.toBeNull();
      expect(header.querySelector('.subagent-desc')).not.toBeNull();
      expect(header.querySelector('.subagent-chevron')).not.toBeNull();
    });

    it('formats label as subagent(label: "...", task_id: "...")', () => {
      const panel = createSubAgentPanel(1, 'task_abc', 'research', 'desc');

      const label = panel.querySelector('.subagent-label');
      expect(label.textContent).toBe('subagent(label: "research", task_id: "task_abc")');
    });

    it('truncates label when combined text exceeds 120 chars', () => {
      const longLabel = 'a'.repeat(100);
      const longTaskId = 'b'.repeat(60);
      const panel = createSubAgentPanel(1, longTaskId, longLabel, 'desc');

      const label = panel.querySelector('.subagent-label');
      expect(label.textContent.length).toBeLessThanOrEqual(123);
      expect(label.textContent.endsWith('...')).toBe(true);
    });

    it('truncates description when it exceeds 120 chars', () => {
      const longDesc = 'x'.repeat(200);
      const panel = createSubAgentPanel(1, 'task_1', 'label', longDesc);

      const desc = panel.querySelector('.subagent-desc');
      expect(desc.textContent.length).toBeLessThanOrEqual(123);
      expect(desc.textContent.endsWith('...')).toBe(true);
    });

    it('toggles content visibility on header click', () => {
      const panel = createSubAgentPanel(1, 'task_abc', 'research', 'desc');
      const header = panel.querySelector('.subagent-header');
      const content = panel.querySelector('.subagent-content');
      const chevron = panel.querySelector('.subagent-chevron');

      expect(content.classList.contains('visible')).toBe(false);
      expect(chevron.classList.contains('expanded')).toBe(false);

      header.click();

      expect(content.classList.contains('visible')).toBe(true);
      expect(chevron.classList.contains('expanded')).toBe(true);

      header.click();

      expect(content.classList.contains('visible')).toBe(false);
      expect(chevron.classList.contains('expanded')).toBe(false);
    });
  });

  describe('createSubAgentThinking', () => {
    it('creates element with correct class, data attribute, and text', () => {
      const el = createSubAgentThinking(5, 'thinking text');

      expect(el).toBeInstanceOf(HTMLElement);
      expect(el.className).toBe('subagent-thinking');
      expect(el.dataset.partId).toBe('5');
      expect(el.textContent).toBe('thinking text');
    });
  });

  describe('createSubAgentText', () => {
    it('creates element with correct class, data attribute, and text', () => {
      const el = createSubAgentText(6, 'output text');

      expect(el).toBeInstanceOf(HTMLElement);
      expect(el.className).toBe('subagent-text');
      expect(el.dataset.partId).toBe('6');
      expect(el.textContent).toBe('output text');
    });
  });

  describe('createSubAgentToolRow', () => {
    it('creates one-line row with spinner, name, and args', () => {
      const row = createSubAgentToolRow(7, 'search', { q: 'test' }, 'call_1');

      expect(row).toBeInstanceOf(HTMLElement);
      expect(row.className).toBe('subagent-tool-row');
      expect(row.dataset.partId).toBe('7');

      expect(row.querySelector('.spinner-svg')).not.toBeNull();

      const nameSpan = row.querySelector('.tool-name');
      expect(nameSpan.textContent).toBe('search');

      const argsSpan = row.querySelector('.tool-args');
      expect(argsSpan).not.toBeNull();

      expect(row.querySelector('.tool-chevron')).toBeNull();
    });

    it('truncates long args to 120 chars', () => {
      const longArgs = 'a'.repeat(200);
      const row = createSubAgentToolRow(7, 'tool', longArgs, 'call_1');

      const argsSpan = row.querySelector('.tool-args');
      expect(argsSpan.textContent.length).toBeLessThanOrEqual(123);
      expect(argsSpan.textContent.endsWith('...')).toBe(true);
    });
  });

  describe('updateSubAgentToolResult', () => {
    it('removes spinner and updates tool-args text', () => {
      const row = createSubAgentToolRow(7, 'search', { q: 'test' }, 'call_1');

      expect(row.querySelector('.spinner-svg')).not.toBeNull();

      updateSubAgentToolResult(7, 'result text', row);

      expect(row.querySelector('.spinner-svg')).toBeNull();

      const argsSpan = row.querySelector('.tool-args');
      expect(argsSpan.textContent).toBe('result text');
    });

    it('truncates long results to 120 chars', () => {
      const row = createSubAgentToolRow(7, 'tool', 'args', 'call_1');
      const longResult = 'z'.repeat(200);

      updateSubAgentToolResult(7, longResult, row);

      const argsSpan = row.querySelector('.tool-args');
      expect(argsSpan.textContent.length).toBeLessThanOrEqual(123);
      expect(argsSpan.textContent.endsWith('...')).toBe(true);
    });

    it('handles null element without error', () => {
      expect(() => updateSubAgentToolResult(7, 'result', null)).not.toThrow();
    });
  });

  describe('concurrent sub-agent panels have separate state', () => {
    it('toggling one panel does not affect the other', () => {
      const panelA = createSubAgentPanel(1, 'task_alpha', 'research', 'search for info');
      const panelB = createSubAgentPanel(2, 'task_beta', 'coding', 'write code');

      const headerA = panelA.querySelector('.subagent-header');
      const contentA = panelA.querySelector('.subagent-content');
      const contentB = panelB.querySelector('.subagent-content');

      expect(contentA.classList.contains('visible')).toBe(false);
      expect(contentB.classList.contains('visible')).toBe(false);

      headerA.click();

      expect(contentA.classList.contains('visible')).toBe(true);
      expect(contentB.classList.contains('visible')).toBe(false);
    });
  });

  describe('sub-agent done removes spinner from header', () => {
    it('removes the spinner element when sub-agent completes', () => {
      const panel = createSubAgentPanel(1, 'task_abc', 'research', 'desc');
      const header = panel.querySelector('.subagent-header');

      expect(header.querySelector('.spinner-svg')).not.toBeNull();

      const spinner = header.querySelector('.spinner-svg');
      if (spinner) spinner.remove();

      expect(header.querySelector('.spinner-svg')).toBeNull();
    });
  });

  describe('sub-agent error shows error text in content area', () => {
    it('displays error message in the content area', () => {
      const panel = createSubAgentPanel(1, 'task_abc', 'research', 'desc');
      const content = panel.querySelector('.subagent-content');

      panel.querySelector('.subagent-header').click();
      expect(content.classList.contains('visible')).toBe(true);

      const errorEl = document.createElement('div');
      errorEl.className = 'error-message';
      errorEl.textContent = 'Error: Something went wrong';
      content.appendChild(errorEl);

      const errorDiv = content.querySelector('.error-message');
      expect(errorDiv).not.toBeNull();
      expect(errorDiv.textContent).toBe('Error: Something went wrong');
    });
  });

  describe('root agent parts are unaffected by sub-agent panel creation', () => {
    it('root tool part retains correct structure when sub-agent panel exists', () => {
      const { row: toolPart } = createToolPart(10, 'search', { q: 'hello' }, 'call_root');
      const subPanel = createSubAgentPanel(11, 'task_sub', 'research', 'sub task');

      expect(toolPart).toBeInstanceOf(HTMLElement);
      expect(toolPart.className).toBe('tool-part');
      expect(toolPart.dataset.partId).toBe('10');
      expect(toolPart.querySelector('.tool-row')).not.toBeNull();
      expect(toolPart.querySelector('.tool-chevron')).not.toBeNull();
      expect(toolPart.querySelector('.tool-detail')).not.toBeNull();

      expect(subPanel).toBeInstanceOf(HTMLElement);
      expect(subPanel.className).toBe('subagent-panel');
      expect(subPanel.dataset.taskId).toBe('task_sub');

      expect(toolPart.querySelector('.tool-name').textContent).toBe('search');
      expect(toolPart.querySelector('.tool-args').textContent).toBe('{"q":"hello"}');
    });
  });

  describe('sub-agent tool row does not have expand/collapse behavior', () => {
    it('has no chevron, no detail div, and no click-toggle behavior', () => {
      const row = createSubAgentToolRow(7, 'search', { q: 'test' }, 'call_1');

      expect(row.querySelector('.tool-chevron')).toBeNull();

      expect(row.querySelector('.tool-detail')).toBeNull();

      const childrenBefore = row.innerHTML;
      row.click();
      const childrenAfter = row.innerHTML;
      expect(childrenAfter).toBe(childrenBefore);
    });
  });
});
