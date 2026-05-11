import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const partsCode = readFileSync(join(__dirname, '../parts.js'), 'utf-8');

function loadParts() {
  const fn = new Function(
    'document', 'marked', 'autoScroll',
    partsCode + '; return { createReasoningPart, createTextPart, updatePartContent, closePart, createToolPart, updateToolResult };'
  );
  return fn(document, globalThis.marked, globalThis.autoScroll);
}

describe('Part rendering functions', () => {
  let fns;

  beforeEach(() => {
    document.body.innerHTML = '';
    document.head.innerHTML = '';
    globalThis.marked = { parse: vi.fn(text => `<p>${text}</p>`) };
    globalThis.autoScroll = vi.fn();
    fns = loadParts();
    globalThis.createReasoningPart = fns.createReasoningPart;
    globalThis.createTextPart = fns.createTextPart;
    globalThis.updatePartContent = fns.updatePartContent;
    globalThis.closePart = fns.closePart;
    globalThis.createToolPart = fns.createToolPart;
    globalThis.updateToolResult = fns.updateToolResult;
  });

  describe('createReasoningPart', () => {
    it('creates a thinking-block element with correct content and data attribute', () => {
      const el = createReasoningPart('part-1', 'thinking about things');

      expect(el).toBeInstanceOf(HTMLElement);
      expect(el.className).toBe('thinking-block');
      expect(el.textContent).toBe('thinking about things');
      expect(el.dataset.partId).toBe('part-1');
    });
  });

  describe('createTextPart', () => {
    it('creates a text-block element with correct content and data attribute', () => {
      const el = createTextPart('part-2', 'hello world');

      expect(el).toBeInstanceOf(HTMLElement);
      expect(el.className).toBe('text-block');
      expect(el.textContent).toBe('hello world');
      expect(el.dataset.partId).toBe('part-2');
    });
  });

  describe('updatePartContent', () => {
    it('sets element textContent to the given content', () => {
      const el = document.createElement('div');
      el.textContent = 'old';

      updatePartContent('part-1', 'new content', 'text', el);

      expect(el.textContent).toBe('new content');
    });

    it('does nothing when element is null', () => {
      expect(() => updatePartContent('part-1', 'text', 'text', null)).not.toThrow();
    });
  });

  describe('closePart', () => {
    it('renders markdown for text type using marked.parse', () => {
      const el = document.createElement('div');
      el.textContent = 'some **bold** text';

      closePart('part-1', 'some **bold** text', 'text', el);

      expect(marked.parse).toHaveBeenCalledWith('some **bold** text');
      expect(el.innerHTML).toBe('<p>some **bold** text</p>');
    });

    it('does nothing when element is null', () => {
      expect(() => closePart('part-1', 'text', 'text', null)).not.toThrow();
    });

    it('does not call marked for non-text type', () => {
      const el = document.createElement('div');
      el.textContent = 'original';

      closePart('part-1', 'content', 'tool', el);

      expect(marked.parse).not.toHaveBeenCalled();
      expect(el.textContent).toBe('original');
    });
  });

  describe('createToolPart', () => {
    it('returns { row: container, detail } with correct structure', () => {
      const result = createToolPart('part-3', 'read_file', { path: '/foo.txt' }, 'call-123');

      expect(result).toHaveProperty('row');
      expect(result).toHaveProperty('detail');

      const container = result.row;
      expect(container.className).toBe('tool-part');
      expect(container.dataset.partId).toBe('part-3');
      expect(container.dataset.callId).toBe('call-123');
    });

    it('contains tool-row and tool-detail as children', () => {
      const { row: container, detail } = createToolPart('p', 'bash', 'ls -la', 'c1');

      const toolRow = container.querySelector('.tool-row');
      expect(toolRow).not.toBeNull();

      const toolDetail = container.querySelector('.tool-detail');
      expect(toolDetail).not.toBeNull();
      expect(toolDetail).toBe(detail);
    });

    it('displays tool name and truncated args', () => {
      const longArgs = 'a'.repeat(200);
      const { row: container } = createToolPart('p', 'search', longArgs, 'c2');

      const nameSpan = container.querySelector('.tool-name');
      expect(nameSpan.textContent).toBe('search');

      const argsSpan = container.querySelector('.tool-args');
      expect(argsSpan.textContent.length).toBeLessThanOrEqual(123); // 120 + '...'
      expect(argsSpan.textContent.endsWith('...')).toBe(true);
    });

    it('includes a spinner SVG element', () => {
      const { row: container } = createToolPart('p', 'tool', 'args', 'c3');

      const spinner = container.querySelector('.spinner-svg');
      expect(spinner).not.toBeNull();
      expect(spinner.tagName.toLowerCase()).toBe('svg');
    });

    it('handles null args and empty callId gracefully', () => {
      const { row: container } = createToolPart('p', 'tool', null, null);

      expect(container.dataset.callId).toBe('');
      const argsSpan = container.querySelector('.tool-args');
      expect(argsSpan.textContent).toBe('');
    });
  });

  describe('updateToolResult', () => {
    it('removes spinner and updates detail text', () => {
      const { row: container, detail } = createToolPart('p', 'tool', 'args', 'c1');

      // Verify spinner exists before update
      expect(container.querySelector('.spinner-svg')).not.toBeNull();

      updateToolResult('p', 'done!', container, detail);

      expect(container.querySelector('.spinner-svg')).toBeNull();
      expect(detail.textContent).toBe('done!');
    });

    it('handles object results by JSON stringifying', () => {
      const { row: container, detail } = createToolPart('p', 'tool', 'a', 'c1');
      const result = { status: 'ok', code: 0 };

      updateToolResult('p', result, container, detail);

      expect(detail.textContent).toBe(JSON.stringify(result, null, 2));
    });

    it('does nothing when container is null', () => {
      expect(() => updateToolResult('p', 'result', null, null)).not.toThrow();
    });
  });

  describe('Part ordering', () => {
    it('elements appear in insertion order in DOM', () => {
      const parent = document.createElement('div');

      const el1 = createReasoningPart('1', 'first');
      const el2 = createTextPart('2', 'second');
      const el3 = createTextPart('3', 'third');

      parent.appendChild(el1);
      parent.appendChild(el2);
      parent.appendChild(el3);

      const children = Array.from(parent.children);
      expect(children[0]).toBe(el1);
      expect(children[1]).toBe(el2);
      expect(children[2]).toBe(el3);
      expect(children.map(c => c.textContent)).toEqual(['first', 'second', 'third']);
    });
  });
});
