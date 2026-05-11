import { describe, it, expect, vi, beforeEach } from 'vitest';
import { readFileSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const appCode = readFileSync(join(__dirname, '../app.js'), 'utf-8');

function setupGlobals() {
  globalThis.marked = { parse: vi.fn(text => `<p>${text}</p>`) };
  const store = {};
  globalThis.localStorage = {
    getItem: vi.fn(key => store[key] || null),
    setItem: vi.fn((key, val) => { store[key] = val; }),
    removeItem: vi.fn(key => { delete store[key]; }),
    clear: vi.fn(() => { Object.keys(store).forEach(k => delete store[k]); }),
  };
  globalThis.fetch = vi.fn(() =>
    Promise.resolve({ ok: true, json: () => Promise.resolve({ sessions: [] }) })
  );
  globalThis.confirm = vi.fn(() => true);
}

function loadAppCode() {
  document.body.innerHTML = `
    <div id="app">
      <aside id="sidebar" class="sidebar">
        <div class="sidebar-header">
          <h2>Sessions</h2>
          <button id="sidebar-close-btn" class="sidebar-toggle">‹</button>
        </div>
        <div id="session-list" class="session-list"></div>
      </aside>
      <main id="main-content">
        <header>
          <button id="sidebar-toggle-btn" class="sidebar-toggle">☰</button>
          <h1>Engine</h1>
          <button id="new-chat-btn">New Chat</button>
        </header>
        <div id="messages"><div class="empty-state">Send a message to start</div></div>
        <div class="input-area">
          <textarea id="chat-input" placeholder="Send a message..." rows="1"></textarea>
          <button id="send-btn">Send</button>
        </div>
      </main>
    </div>
  `;

  // Use indirection wrappers so tests can reassign globalThis.fetch/confirm
  // and the app code inside new Function will see the updated behavior
  const fetchProxy = (...args) => globalThis.fetch(...args);
  const confirmProxy = (msg) => globalThis.confirm(msg);

  const fn = new Function(
    'document', 'window', 'localStorage', 'marked', 'fetch', 'confirm',
    appCode + '; return { toggleSidebar, fetchSessions, renderSessionList, resetChatState, hydrateSessionMessages, updateActiveSessionHighlight, switchToSession, deleteSession, updateInputState };'
  );
  return fn(document, window, globalThis.localStorage, globalThis.marked, fetchProxy, confirmProxy);
}

describe('Session sidebar', () => {
  let fns;

  beforeEach(() => {
    document.body.innerHTML = '';
    document.head.innerHTML = '';
    vi.restoreAllMocks();
    setupGlobals();
    fns = loadAppCode();
  });

  describe('toggleSidebar', () => {
    it('adds .open class on first call', () => {
      const sidebar = document.getElementById('sidebar');
      expect(sidebar.classList.contains('open')).toBe(false);

      fns.toggleSidebar();

      expect(sidebar.classList.contains('open')).toBe(true);
    });

    it('removes .open class on second call', () => {
      const sidebar = document.getElementById('sidebar');

      fns.toggleSidebar();
      expect(sidebar.classList.contains('open')).toBe(true);

      fns.toggleSidebar();
      expect(sidebar.classList.contains('open')).toBe(false);
    });
  });

  describe('renderSessionList', () => {
    it('renders two session items with correct titles', () => {
      const sessions = [
        { id: 's1', title: 'Hello world', last_active: '2026-05-11T00:00:00', message_count: 5 },
        { id: 's2', title: 'Test session', last_active: null, message_count: 0 },
      ];

      fns.renderSessionList(sessions);

      const items = document.querySelectorAll('.session-item');
      expect(items.length).toBe(2);
      expect(items[0].querySelector('.session-item-title').textContent).toBe('Hello world');
      expect(items[1].querySelector('.session-item-title').textContent).toBe('Test session');
    });

    it('marks active session with .active class', () => {
      globalThis.localStorage.setItem('engine_session_id', 's1');
      const sessions = [
        { id: 's1', title: 'Active one', last_active: '2026-05-11T00:00:00', message_count: 3 },
        { id: 's2', title: 'Inactive', last_active: null, message_count: 0 },
      ];

      fns.renderSessionList(sessions);

      const items = document.querySelectorAll('.session-item');
      expect(items[0].classList.contains('active')).toBe(true);
      expect(items[1].classList.contains('active')).toBe(false);
    });
  });

  describe('empty state', () => {
    it('shows "No sessions yet" when list is empty', () => {
      fns.renderSessionList([]);

      const empty = document.querySelector('.sidebar-empty');
      expect(empty).not.toBeNull();
      expect(empty.textContent).toBe('No sessions yet');
    });

    it('shows "No sessions yet" when list is null', () => {
      fns.renderSessionList(null);

      const empty = document.querySelector('.sidebar-empty');
      expect(empty).not.toBeNull();
      expect(empty.textContent).toBe('No sessions yet');
    });
  });

  describe('switchToSession', () => {
    it('does not fetch when switching to the already active session', async () => {
      globalThis.localStorage.setItem('engine_session_id', 's1');
      globalThis.fetch = vi.fn(() =>
        Promise.resolve({ ok: true, json: () => Promise.resolve({ sessions: [] }) })
      );

      await fns.switchToSession('s1');

      expect(globalThis.fetch).not.toHaveBeenCalled();
    });

    it('hydrates session when switching to a different session', async () => {
      globalThis.localStorage.setItem('engine_session_id', 's1');
      globalThis.fetch = vi.fn((url, opts) => {
        if (typeof url === 'string' && url.includes('/api/sessions/') && !url.endsWith('/api/sessions')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              messages: [
                { role: 'user', content: 'Hello' },
                { role: 'assistant', content: 'Hi there' },
              ],
            }),
          });
        }
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ sessions: [] }),
        });
      });

      await fns.switchToSession('s2');

      expect(globalThis.fetch).toHaveBeenCalled();
      const fetchCalls = globalThis.fetch.mock.calls.map(c => c[0]);
      expect(fetchCalls.some(u => typeof u === 'string' && u.includes('/api/sessions/s2'))).toBe(true);
    });
  });

  describe('resetChatState', () => {
    it('resets messages to empty state', () => {
      const messagesEl = document.getElementById('messages');
      messagesEl.innerHTML = '<div class="message user"><div class="bubble">Hello</div></div>';
      expect(messagesEl.querySelector('.empty-state')).toBeNull();

      fns.resetChatState();

      expect(messagesEl.querySelector('.empty-state')).not.toBeNull();
      expect(messagesEl.querySelector('.empty-state').textContent).toBe('Send a message to start');
    });

    it('resets send button to idle state', () => {
      const sendBtn = document.getElementById('send-btn');
      sendBtn.textContent = 'Stop';
      sendBtn.classList.add('stop-btn');

      fns.resetChatState();

      expect(sendBtn.textContent).toBe('Send');
      expect(sendBtn.classList.contains('stop-btn')).toBe(false);
    });
  });

  describe('deleteSession', () => {
    it('removes the session DOM element on successful delete', async () => {
      fns.renderSessionList([
        { id: 's1', title: 'To delete', last_active: null, message_count: 0 },
        { id: 's2', title: 'Keep', last_active: null, message_count: 0 },
      ]);
      expect(document.querySelectorAll('.session-item').length).toBe(2);

      globalThis.fetch = vi.fn(() =>
        Promise.resolve({ ok: true, status: 200 })
      );

      await fns.deleteSession('s1');

      expect(document.querySelectorAll('.session-item').length).toBe(1);
      expect(document.querySelector('.session-item[data-session-id="s1"]')).toBeNull();
      expect(document.querySelector('.session-item[data-session-id="s2"]')).not.toBeNull();
    });

    it('aborts when confirm is cancelled', async () => {
      fns.renderSessionList([
        { id: 's1', title: 'Session', last_active: null, message_count: 0 },
      ]);

      globalThis.confirm = vi.fn(() => false);
      globalThis.fetch = vi.fn();

      await fns.deleteSession('s1');

      expect(globalThis.fetch).not.toHaveBeenCalled();
      expect(document.querySelectorAll('.session-item').length).toBe(1);
    });

    it('shows empty state when last session is deleted', async () => {
      fns.renderSessionList([
        { id: 's1', title: 'Only one', last_active: null, message_count: 0 },
      ]);

      globalThis.fetch = vi.fn(() =>
        Promise.resolve({ ok: true, status: 200 })
      );

      await fns.deleteSession('s1');

      const empty = document.querySelector('.sidebar-empty');
      expect(empty).not.toBeNull();
      expect(empty.textContent).toBe('No sessions yet');
    });

    it('clears localStorage and resets chat when deleting the active session', async () => {
      globalThis.localStorage.setItem('engine_session_id', 's1');

      fns.renderSessionList([
        { id: 's1', title: 'Active', last_active: null, message_count: 0 },
      ]);

      globalThis.fetch = vi.fn(() =>
        Promise.resolve({ ok: true, status: 200 })
      );

      await fns.deleteSession('s1');

      expect(globalThis.localStorage.removeItem).toHaveBeenCalledWith('engine_session_id');
      expect(document.getElementById('messages').querySelector('.empty-state')).not.toBeNull();
    });

    it('does not clear localStorage when deleting an inactive session', async () => {
      globalThis.localStorage.setItem('engine_session_id', 's2');

      fns.renderSessionList([
        { id: 's1', title: 'Other', last_active: null, message_count: 0 },
        { id: 's2', title: 'Active', last_active: null, message_count: 0 },
      ]);

      globalThis.fetch = vi.fn(() =>
        Promise.resolve({ ok: true, status: 200 })
      );

      await fns.deleteSession('s1');

      expect(globalThis.localStorage.removeItem).not.toHaveBeenCalledWith('engine_session_id');
    });
  });
});
