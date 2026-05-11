        let agentState = 'idle'; // 'idle' | 'agent_running' | 'waiting_for_children'
        let parts = [];
        let subagents = {};
        let autoScrollEnabled = true;
        let activeContentStream = null; // Current assistant message content container
        const SESSION_KEY = 'engine_session_id';
        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const newChatBtn = document.getElementById('new-chat-btn');
        const sidebarEl = document.getElementById('sidebar');
        const sessionListEl = document.getElementById('session-list');
        const sidebarToggleBtn = document.getElementById('sidebar-toggle-btn');
        const sidebarCloseBtn = document.getElementById('sidebar-close-btn');

        inputEl.addEventListener('input', () => {
            inputEl.style.height = 'auto';
            inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
        });

        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (inputEl.value.trim()) {
                    sendMessage();
                }
            }
        });

        sendBtn.addEventListener('click', () => {
            if (agentState !== 'idle') {
                abortAgent();
                return;
            }
            if (inputEl.value.trim()) {
                sendMessage();
            }
        });

        newChatBtn.addEventListener('click', () => {
            if (agentState !== 'idle') {
                abortAgent();
            }
            localStorage.removeItem(SESSION_KEY);
            messagesEl.innerHTML = '<div class="empty-state">Send a message to start</div>';
            // Close sidebar if open
            if (sidebarEl.classList.contains('open')) {
                sidebarEl.classList.remove('open');
            }
        });

        messagesEl.addEventListener('scroll', () => {
            const threshold = 80;
            const distanceFromBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight;
            autoScrollEnabled = distanceFromBottom < threshold;
        });

        function autoScroll() {
            if (autoScrollEnabled) {
                messagesEl.scrollTop = messagesEl.scrollHeight;
            }
        }

        function toggleSidebar() {
            sidebarEl.classList.toggle('open');
        }

        sidebarToggleBtn.addEventListener('click', toggleSidebar);
        sidebarCloseBtn.addEventListener('click', toggleSidebar);

        function getSessionId() {
            return localStorage.getItem(SESSION_KEY);
        }

        function setSessionId(id) {
            if (id) {
                localStorage.setItem(SESSION_KEY, id);
            }
        }

        function clearEmptyState() {
            const empty = messagesEl.querySelector('.empty-state');
            if (empty) empty.remove();
        }

        function appendUserMessage(text) {
            clearEmptyState();
            const wrapper = document.createElement('div');
            wrapper.className = 'message user';
            const bubble = document.createElement('div');
            bubble.className = 'bubble';
            bubble.textContent = text;
            wrapper.appendChild(bubble);
            messagesEl.appendChild(wrapper);
            autoScroll();
        }

        function resetChatState() {
            messagesEl.innerHTML = '<div class="empty-state">Send a message to start</div>';
            parts = [];
            subagents = {};
            activeContentStream = null;
            agentState = 'idle';
            updateInputState();
        }

        async function hydrateSessionMessages(sessionId) {
            try {
                const response = await fetch('/api/sessions/' + sessionId);
                if (!response.ok) throw new Error('Failed to load session');
                const data = await response.json();

                parts = [];
                subagents = {};
                activeContentStream = null;
                agentState = 'idle';
                updateInputState();

                messagesEl.innerHTML = '';

                const messages = data.messages || [];
                let hasMessages = false;
                for (const msg of messages) {
                    if (msg.role === 'system' || msg.role === 'tool') continue;
                    hasMessages = true;
                    if (msg.role === 'user') {
                        appendUserMessage(msg.content);
                    } else if (msg.role === 'assistant') {
                        const wrapper = document.createElement('div');
                        wrapper.className = 'message assistant';
                        const textBlock = document.createElement('div');
                        textBlock.className = 'text-block';
                        textBlock.innerHTML = typeof marked !== 'undefined' ? marked.parse(msg.content || '') : (msg.content || '');
                        wrapper.appendChild(textBlock);
                        messagesEl.appendChild(wrapper);
                    }
                }

                if (!hasMessages) {
                    messagesEl.innerHTML = '<div class="empty-state">Send a message to start</div>';
                }

                setSessionId(sessionId);

                updateActiveSessionHighlight(sessionId);

            } catch (err) {
                console.error('Failed to hydrate session:', err);
                resetChatState();
            }
        }

        function updateActiveSessionHighlight(sessionId) {
            const items = document.querySelectorAll('.session-item');
            items.forEach(item => {
                if (item.dataset.sessionId === sessionId) {
                    item.classList.add('active');
                } else {
                    item.classList.remove('active');
                }
            });
        }

        function createAssistantMessage() {
            clearEmptyState();
            parts = [];
            const wrapper = document.createElement('div');
            wrapper.className = 'message assistant';
            const contentStream = document.createElement('div');
            contentStream.className = 'content-stream';
            wrapper.appendChild(contentStream);
            messagesEl.appendChild(wrapper);
            autoScroll();
            return { wrapper, contentStream };
        }

        function updateInputState() {
            inputEl.classList.remove('agent-running');
            if (agentState !== 'idle') {
                inputEl.disabled = true;
                sendBtn.disabled = false;
                sendBtn.textContent = 'Stop';
                sendBtn.classList.add('stop-btn');
                inputEl.classList.add('agent-running');
            } else {
                inputEl.disabled = false;
                sendBtn.disabled = false;
                sendBtn.textContent = 'Send';
                sendBtn.classList.remove('stop-btn');
            }
        }

        async function abortAgent() {
            if (agentState === 'idle') return;
            agentState = 'idle';
            updateInputState();
            if (window._activeReader) {
                try { await window._activeReader.cancel(); } catch (e) {}
                window._activeReader = null;
            }
            for (const part of parts) {
                if (part.state === 'open') {
                    part.state = 'closed';
                    if (part.type === 'tool' && part.element) {
                        const spinner = part.element.querySelector('.spinner-svg');
                        if (spinner) spinner.remove();
                    }
                }
            }
            for (const id in subagents) {
                const sa = subagents[id];
                if (sa.state === 'running') {
                    sa.state = 'aborted';
                    const spinner = sa.element.querySelector('.subagent-header .spinner-svg');
                    if (spinner) spinner.remove();
                    sa.contentEl.querySelectorAll('.spinner-svg').forEach(s => s.remove());
                }
            }
            try {
                await fetch('/api/chat/abort', { method: 'POST' });
            } catch (e) {}
        }


        function handleSSEEvent(eventType, data) {
            switch (eventType) {
                case 'agent_start': {
                    if (data.session_id) setSessionId(data.session_id);
                    break;
                }
                case 'part_new': {
                    const part = {
                        id: data.part_id,
                        type: data.part_type,
                        content: data.text || '',
                        element: null,
                        state: 'open',
                    };
                    if (data.part_type === 'reasoning') {
                        part.element = createReasoningPart(data.part_id, data.text || '');
                    } else if (data.part_type === 'text') {
                        part.element = createTextPart(data.part_id, data.text || '');
                    }
                    parts.push(part);
                    if (part.element) {
                        activeContentStream.appendChild(part.element);
                    }
                    autoScroll();
                    break;
                }
                case 'part_delta': {
                    const part = parts.find(p => p.id === data.part_id);
                    if (part) {
                        part.content += data.text || '';
                        updatePartContent(data.part_id, part.content, part.type, part.element);
                    }
                    autoScroll();
                    break;
                }
                case 'part_close': {
                    const part = parts.find(p => p.id === data.part_id);
                    if (part) {
                        part.state = 'closed';
                        closePart(data.part_id, part.content, part.type, part.element);
                    }
                    autoScroll();
                    break;
                }
                case 'tool_call_start': {
                    const part = {
                        id: data.part_id,
                        type: 'tool',
                        content: '',
                        toolName: data.tool_name,
                        callId: data.call_id,
                        element: null,
                        state: 'open',
                    };
                    const { row, detail } = createToolPart(data.part_id, data.tool_name, data.arguments, data.call_id);
                    part.element = row;
                    part.detail = detail;
                    parts.push(part);
                    activeContentStream.appendChild(row);
                    autoScroll();
                    break;
                }
                case 'tool_call_result': {
                    const part = parts.find(p => p.id === data.part_id);
                    if (part) {
                        updateToolResult(data.part_id, data.result, part.element, part.detail);
                    }
                    autoScroll();
                    break;
                }
                case 'done': {
                    agentState = 'idle';
                    updateInputState();
                    if (data.session_id) setSessionId(data.session_id);
                    for (const part of parts) {
                        if (part.state === 'open') {
                            part.state = 'closed';
                            if (part.type === 'text') {
                                closePart(part.id, part.content, part.type, part.element);
                            }
                        }
                    }
                    break;
                }
                case 'error': {
                    agentState = 'idle';
                    updateInputState();
                    for (const part of parts) {
                        if (part.state === 'open') {
                            part.state = 'closed';
                        }
                    }
                    const errorEl = document.createElement('div');
                    errorEl.className = 'error-message';
                    errorEl.textContent = 'Error: ' + (data.message || 'Unknown error');
                    activeContentStream.appendChild(errorEl);
                    autoScroll();
                    break;
                }
                case 'turn_start': {
                    const { contentStream } = createAssistantMessage();
                    activeContentStream = contentStream;
                    break;
                }
                case 'waiting_for_children': {
                    agentState = 'waiting_for_children';
                    updateInputState();
                    break;
                }
                case 'subagent_start': {
                    const panel = createSubAgentPanel(data.part_id, data.task_id, data.label, data.description);
                    subagents[data.task_id] = {
                        part_id: data.part_id,
                        element: panel,
                        contentEl: panel.querySelector('.subagent-content'),
                        parts: [],
                        state: 'running',
                    };
                    activeContentStream.appendChild(panel);
                    autoScroll();
                    break;
                }
                case 'subagent_part_new': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    let el;
                    if (data.part_type === 'reasoning') {
                        el = createSubAgentThinking(data.part_id, data.text || '');
                    } else if (data.part_type === 'text') {
                        el = createSubAgentText(data.part_id, data.text || '');
                    }
                    if (el) {
                        sa.parts.push({ id: data.part_id, type: data.part_type, element: el, content: data.text || '' });
                        sa.contentEl.appendChild(el);
                    }
                    autoScroll();
                    break;
                }
                case 'subagent_part_delta': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    const part = sa.parts.find(p => p.id === data.part_id);
                    if (part && part.element) {
                        part.content += data.text || '';
                        part.element.textContent = part.content;
                    }
                    autoScroll();
                    break;
                }
                case 'subagent_part_close': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    const part = sa.parts.find(p => p.id === data.part_id);
                    if (part) {
                        part.state = 'closed';
                        if (part.type === 'reasoning' && part.element) {
                            part.content = part.content.trimEnd();
                            part.element.textContent = part.content;
                        }
                        if (part.type === 'text' && part.element && part.content && typeof marked !== 'undefined') {
                            part.element.innerHTML = marked.parse(part.content);
                        }
                    }
                    break;
                }
                case 'subagent_tool_start': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    const toolRow = createSubAgentToolRow(data.part_id, data.tool_name, data.arguments, data.call_id);
                    sa.parts.push({ id: data.part_id, type: 'tool', element: toolRow, state: 'open' });
                    sa.contentEl.appendChild(toolRow);
                    autoScroll();
                    break;
                }
                case 'subagent_tool_result': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    const part = sa.parts.find(p => p.id === data.part_id);
                    if (part && part.element) {
                        const res = updateSubAgentToolResult(data.part_id, data.result, data.tool_name, part.element);
                        if (res === 'removed') {
                            sa.parts = sa.parts.filter(p => p.id !== data.part_id);
                        }
                    }
                    autoScroll();
                    break;
                }
                case 'subagent_done': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    sa.state = 'completed';
                    const spinner = sa.element.querySelector('.subagent-header .spinner-svg');
                    if (spinner) spinner.remove();
                    break;
                }
                case 'subagent_error': {
                    const sa = subagents[data.task_id];
                    if (!sa) break;
                    sa.state = 'error';
                    const spinner = sa.element.querySelector('.subagent-header .spinner-svg');
                    if (spinner) spinner.remove();
                    const errorEl = document.createElement('div');
                    errorEl.className = 'error-message';
                    errorEl.textContent = 'Error: ' + (data.message || 'Unknown error');
                    sa.contentEl.appendChild(errorEl);
                    autoScroll();
                    break;
                }
            }
        }

        function formatRelativeTime(isoString) {
            try {
                const date = new Date(isoString);
                const now = new Date();
                const diffMs = now - date;
                const diffMins = Math.floor(diffMs / 60000);
                const diffHours = Math.floor(diffMs / 3600000);
                const diffDays = Math.floor(diffMs / 86400000);
                if (diffMins < 1) return 'just now';
                if (diffMins < 60) return diffMins + 'm ago';
                if (diffHours < 24) return diffHours + 'h ago';
                if (diffDays < 7) return diffDays + 'd ago';
                return date.toLocaleDateString();
            } catch (e) {
                return '';
            }
        }

        async function fetchSessions() {
            sessionListEl.innerHTML = '<div class="sidebar-loading">Loading sessions...</div>';
            try {
                const response = await fetch('/api/sessions');
                if (!response.ok) throw new Error('Failed to fetch');
                const data = await response.json();
                renderSessionList(data.sessions);
            } catch (err) {
                sessionListEl.innerHTML = '<div class="sidebar-error"><span>Failed to load sessions</span><button class="sidebar-error-retry" onclick="fetchSessions()">Retry</button></div>';
            }
        }

        function renderSessionList(sessions) {
            sessionListEl.innerHTML = '';
            if (!sessions || sessions.length === 0) {
                sessionListEl.innerHTML = '<div class="sidebar-empty">No sessions yet</div>';
                return;
            }
            const activeId = localStorage.getItem(SESSION_KEY);
            sessions.forEach(session => {
                const item = document.createElement('div');
                item.className = 'session-item' + (session.id === activeId ? ' active' : '');
                item.dataset.sessionId = session.id;

                const info = document.createElement('div');
                info.className = 'session-item-info';

                const title = document.createElement('div');
                title.className = 'session-item-title';
                title.textContent = session.title || 'New Session';

                const meta = document.createElement('div');
                meta.className = 'session-item-meta';
                const count = session.message_count || 0;
                const timeStr = session.last_active ? formatRelativeTime(session.last_active) : '';
                meta.textContent = timeStr + (timeStr && count ? ' · ' : '') + count + ' msg' + (count !== 1 ? 's' : '');

                info.appendChild(title);
                info.appendChild(meta);

                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'delete-btn';
                deleteBtn.textContent = '×';
                deleteBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    if (typeof deleteSession === 'function') {
                        deleteSession(session.id);
                    }
                });

                item.appendChild(info);
                item.appendChild(deleteBtn);

                item.addEventListener('click', () => {
                    if (typeof switchToSession === 'function') {
                        switchToSession(session.id);
                    }
                });

                sessionListEl.appendChild(item);
            });
        }

        let _lastSwitchTime = 0;

        async function switchToSession(sessionId) {
            const now = Date.now();
            if (now - _lastSwitchTime < 300) return;
            _lastSwitchTime = now;

            const activeId = localStorage.getItem(SESSION_KEY);
            if (sessionId === activeId) return;

            if (agentState !== 'idle') {
                await abortAgent();
            }

            await hydrateSessionMessages(sessionId);
            fetchSessions();
        }

        async function deleteSession(sessionId) {
            if (!confirm('Delete this session?')) return;

            try {
                const response = await fetch('/api/sessions/' + sessionId, { method: 'DELETE' });
                if (!response.ok && response.status !== 404) {
                    console.error('Failed to delete session:', response.status);
                    return;
                }
            } catch (err) {
                console.error('Failed to delete session:', err);
                return;
            }

            const activeId = localStorage.getItem(SESSION_KEY);
            if (sessionId === activeId) {
                localStorage.removeItem(SESSION_KEY);
                resetChatState();
            }

            const item = document.querySelector('.session-item[data-session-id="' + sessionId + '"]');
            if (item) item.remove();

            if (sessionListEl.children.length === 0) {
                sessionListEl.innerHTML = '<div class="sidebar-empty">No sessions yet</div>';
            }
        }

        async function sendMessage() {
            const message = inputEl.value.trim();
            if (!message) return;

            if (agentState !== 'idle') return;

            agentState = 'agent_running';
            updateInputState();
            inputEl.value = '';
            inputEl.style.height = 'auto';

            appendUserMessage(message);

            const { contentStream } = createAssistantMessage();
            activeContentStream = contentStream;

            try {
                const body = { message };
                const sessionId = getSessionId();
                if (sessionId) {
                    body.session_id = sessionId;
                }

                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });

                if (!response.ok) {
                    if (response.status === 429) {
                        agentState = 'idle';
                        updateInputState();
                    }
                    const errText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errText}`);
                }

                const reader = response.body.getReader();
                window._activeReader = reader;
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    let eventType = '';
                    for (let line of lines) {
                        line = line.trimEnd();
                        if (!line) continue;
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7);
                        } else if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                handleSSEEvent(eventType, data);
                            } catch (e) {
                                console.warn('Failed to parse SSE data:', line, e);
                            }
                        }
                    }
                }

                if (buffer.trim()) {
                    const remaining = buffer.split('\n');
                    let eventType = '';
                    for (let line of remaining) {
                        line = line.trimEnd();
                        if (!line) continue;
                        if (line.startsWith('event: ')) {
                            eventType = line.slice(7);
                        } else if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                handleSSEEvent(eventType, data);
                            } catch (e) {
                            }
                        }
                    }
                }

            } catch (err) {
                const errorEl = document.createElement('div');
                errorEl.className = 'error-message';
                errorEl.textContent = 'Error: ' + err.message;
                activeContentStream.appendChild(errorEl);
            } finally {
                window._activeReader = null;
                agentState = 'idle';
                updateInputState();
                inputEl.focus();
            }
        }

        inputEl.focus();
        fetchSessions();
