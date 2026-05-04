function createReasoningPart(partId, text) {
    const el = document.createElement('div');
    el.className = 'thinking-block';
    el.dataset.partId = partId;
    el.textContent = text;
    return el;
}

function createTextPart(partId, text) {
    const el = document.createElement('div');
    el.className = 'text-block';
    el.dataset.partId = partId;
    el.textContent = text;
    return el;
}

function updatePartContent(partId, content, type, element) {
    if (!element) return;
    element.textContent = content;
}

function closePart(partId, content, type, element) {
    if (!element) return;
    if (type === 'text' && content && typeof marked !== 'undefined') {
        element.innerHTML = marked.parse(content);
    }
}

function createToolPart(partId, toolName, args, callId) {
    const container = document.createElement('div');
    container.className = 'tool-part';
    container.dataset.partId = partId;
    container.dataset.callId = callId || '';

    const row = document.createElement('div');
    row.className = 'tool-row';

    const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    spinner.setAttribute('class', 'spinner-svg');
    spinner.setAttribute('viewBox', '0 0 24 24');
    spinner.setAttribute('fill', 'none');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '12');
    circle.setAttribute('cy', '12');
    circle.setAttribute('r', '10');
    circle.setAttribute('stroke', 'currentColor');
    circle.setAttribute('stroke-width', '2');
    circle.setAttribute('stroke-dasharray', '31.4 31.4');
    circle.setAttribute('stroke-linecap', 'round');
    spinner.appendChild(circle);
    spinner.style.animation = 'spin 1s linear infinite';

    if (!document.getElementById('spin-keyframes')) {
        const style = document.createElement('style');
        style.id = 'spin-keyframes';
        style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }

    const nameSpan = document.createElement('span');
    nameSpan.className = 'tool-name';
    nameSpan.textContent = toolName;

    const argsSpan = document.createElement('span');
    argsSpan.className = 'tool-args';
    argsSpan.textContent = typeof args === 'string' ? args : (args ? JSON.stringify(args) : '');
    if (argsSpan.textContent.length > 120) {
        argsSpan.textContent = argsSpan.textContent.substring(0, 120) + '...';
    }

    const chevron = document.createElement('span');
    chevron.className = 'tool-chevron';
    chevron.textContent = '\u25B6';

    row.appendChild(spinner);
    row.appendChild(nameSpan);
    row.appendChild(argsSpan);
    row.appendChild(chevron);

    const detail = document.createElement('div');
    detail.className = 'tool-detail';
    if (args) {
        detail.textContent = typeof args === 'string' ? args : JSON.stringify(args, null, 2);
    }

    row.addEventListener('click', () => {
        const isVisible = detail.classList.contains('visible');
        if (isVisible) {
            detail.classList.remove('visible');
            chevron.classList.remove('expanded');
        } else {
            detail.classList.add('visible');
            chevron.classList.add('expanded');
        }
        if (typeof autoScroll === 'function') autoScroll();
    });

    container.appendChild(row);
    container.appendChild(detail);

    return { row: container, detail };
}

function updateToolResult(partId, result, container, detail) {
    if (!container) return;
    const spinner = container.querySelector('.spinner-svg');
    if (spinner) spinner.remove();

    if (detail && result) {
        detail.textContent = typeof result === 'string' ? result : JSON.stringify(result, null, 2);
    }
}

function createSubAgentPanel(partId, taskId, label, description) {
    const panel = document.createElement('div');
    panel.className = 'subagent-panel';
    panel.dataset.partId = partId;
    panel.dataset.taskId = taskId;

    const header = document.createElement('div');
    header.className = 'subagent-header';

    const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    spinner.setAttribute('class', 'spinner-svg');
    spinner.setAttribute('viewBox', '0 0 24 24');
    spinner.setAttribute('fill', 'none');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '12');
    circle.setAttribute('cy', '12');
    circle.setAttribute('r', '10');
    circle.setAttribute('stroke', 'currentColor');
    circle.setAttribute('stroke-width', '2');
    circle.setAttribute('stroke-dasharray', '31.4 31.4');
    circle.setAttribute('stroke-linecap', 'round');
    spinner.appendChild(circle);
    spinner.style.animation = 'spin 1s linear infinite';

    if (!document.getElementById('spin-keyframes')) {
        const style = document.createElement('style');
        style.id = 'spin-keyframes';
        style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }

    const labelText = 'subagent(label: "' + (label || 'unknown') + '", task_id: "' + taskId + '")';
    const labelSpan = document.createElement('span');
    labelSpan.className = 'subagent-label';
    labelSpan.textContent = labelText.length > 120 ? labelText.substring(0, 120) + '...' : labelText;

    const descText = description || '';
    const descSpan = document.createElement('span');
    descSpan.className = 'subagent-desc';
    descSpan.textContent = descText.length > 120 ? descText.substring(0, 120) + '...' : descText;

    const chevron = document.createElement('span');
    chevron.className = 'subagent-chevron';
    chevron.textContent = '\u25B6';

    header.appendChild(spinner);
    header.appendChild(labelSpan);
    header.appendChild(descSpan);
    header.appendChild(chevron);

    const content = document.createElement('div');
    content.className = 'subagent-content';

    header.addEventListener('click', () => {
        const isVisible = content.classList.contains('visible');
        if (isVisible) {
            content.classList.remove('visible');
            chevron.classList.remove('expanded');
        } else {
            content.classList.add('visible');
            chevron.classList.add('expanded');
        }
        if (typeof autoScroll === 'function') autoScroll();
    });

    panel.appendChild(header);
    panel.appendChild(content);

    return panel;
}

function createSubAgentThinking(partId, text) {
    const el = document.createElement('div');
    el.className = 'subagent-thinking';
    el.dataset.partId = partId;
    el.textContent = text;
    return el;
}

function createSubAgentText(partId, text) {
    const el = document.createElement('div');
    el.className = 'subagent-text';
    el.dataset.partId = partId;
    el.textContent = text;
    return el;
}

function createSubAgentToolRow(partId, toolName, args, callId) {
    const row = document.createElement('div');
    row.className = 'subagent-tool-row';
    row.dataset.partId = partId;
    row.dataset.callId = callId || '';

    const spinner = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    spinner.setAttribute('class', 'spinner-svg');
    spinner.setAttribute('viewBox', '0 0 24 24');
    spinner.setAttribute('fill', 'none');
    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    circle.setAttribute('cx', '12');
    circle.setAttribute('cy', '12');
    circle.setAttribute('r', '10');
    circle.setAttribute('stroke', 'currentColor');
    circle.setAttribute('stroke-width', '2');
    circle.setAttribute('stroke-dasharray', '31.4 31.4');
    circle.setAttribute('stroke-linecap', 'round');
    spinner.appendChild(circle);
    spinner.style.animation = 'spin 1s linear infinite';

    if (!document.getElementById('spin-keyframes')) {
        const style = document.createElement('style');
        style.id = 'spin-keyframes';
        style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
        document.head.appendChild(style);
    }

    const nameSpan = document.createElement('span');
    nameSpan.className = 'tool-name';
    nameSpan.textContent = toolName;

    const argsStr = typeof args === 'string' ? args : (args ? JSON.stringify(args) : '');
    const argsSpan = document.createElement('span');
    argsSpan.className = 'tool-args';
    argsSpan.textContent = argsStr.length > 120 ? argsStr.substring(0, 120) + '...' : argsStr;

    row.appendChild(spinner);
    row.appendChild(nameSpan);
    row.appendChild(argsSpan);

    return row;
}

function updateSubAgentToolResult(partId, result, element) {
    if (!element) return;
    const spinner = element.querySelector('.spinner-svg');
    if (spinner) spinner.remove();

    if (result) {
        const resultStr = typeof result === 'string' ? result : JSON.stringify(result);
        const argsSpan = element.querySelector('.tool-args');
        if (argsSpan) {
            const truncated = resultStr.length > 120 ? resultStr.substring(0, 120) + '...' : resultStr;
            argsSpan.textContent = truncated;
        }
    }
}
