        let streaming = false;
        let parts = [];
        let subagents = {};
        let autoScrollEnabled = true;
        const SESSION_KEY = 'engine_session_id';

        const messagesEl = document.getElementById('messages');
        const inputEl = document.getElementById('chat-input');
        const sendBtn = document.getElementById('send-btn');
        const newChatBtn = document.getElementById('new-chat-btn');

        inputEl.addEventListener('input', () => {
            inputEl.style.height = 'auto';
            inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + 'px';
        });

        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!streaming && inputEl.value.trim()) {
                    sendMessage();
                }
            }
        });

        sendBtn.addEventListener('click', () => {
            if (!streaming && inputEl.value.trim()) {
                sendMessage();
            }
        });

        newChatBtn.addEventListener('click', () => {
            if (streaming) return;
            localStorage.removeItem(SESSION_KEY);
            messagesEl.innerHTML = '<div class="empty-state">Send a message to start</div>';
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

        function truncateArgs(args) {
            if (!args) return '';
            const str = typeof args === 'string' ? args : JSON.stringify(args);
            return str.length > 120 ? str.substring(0, 120) + '...' : str;
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

        function handleSSEEvent(eventType, data, contentStream) {
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
                        contentStream.appendChild(part.element);
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
                    contentStream.appendChild(row);
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
                    for (const part of parts) {
                        if (part.state === 'open') {
                            part.state = 'closed';
                        }
                    }
                    const errorEl = document.createElement('div');
                    errorEl.className = 'error-message';
                    errorEl.textContent = 'Error: ' + (data.message || 'Unknown error');
                    contentStream.appendChild(errorEl);
                    autoScroll();
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
                    contentStream.appendChild(panel);
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

        async function sendMessage() {
            const message = inputEl.value.trim();
            if (!message || streaming) return;

            streaming = true;
            sendBtn.disabled = true;
            inputEl.value = '';
            inputEl.style.height = 'auto';

            appendUserMessage(message);

            const { contentStream } = createAssistantMessage();

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
                    const errText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errText}`);
                }

                const reader = response.body.getReader();
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
                                handleSSEEvent(eventType, data, contentStream);
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
                                handleSSEEvent(eventType, data, contentStream);
                            } catch (e) {
                                // skip
                            }
                        }
                    }
                }

            } catch (err) {
                const errorEl = document.createElement('div');
                errorEl.className = 'error-message';
                errorEl.textContent = 'Error: ' + err.message;
                contentStream.appendChild(errorEl);
            } finally {
                streaming = false;
                sendBtn.disabled = false;
                inputEl.focus();
            }
        }

        inputEl.focus();
