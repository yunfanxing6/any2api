(() => {
  const VERIFY_ENDPOINT = '/webui/api/verify';
  const MODELS_ENDPOINT = '/webui/api/models';
  const GENERATE_ENDPOINT = '/webui/api/images/generations';
  const EDIT_ENDPOINT = '/webui/api/images/edits';
  const STORAGE_KEY = 'any2api:webui:image-studio:v1';
  const MAX_HISTORY = 12;

  const promptInput = document.getElementById('promptInput');
  const sendBtn = document.getElementById('sendBtn');
  const uploadBtn = document.getElementById('uploadBtn');
  const referenceInput = document.getElementById('referenceInput');
  const referenceBar = document.getElementById('referenceBar');
  const sessionList = document.getElementById('sessionList');
  const feed = document.getElementById('feed');
  const emptyState = document.getElementById('emptyState');
  const newConversationBtn = document.getElementById('newConversationBtn');
  const modelSelect = document.getElementById('modelSelect');
  const countSelect = document.getElementById('countSelect');
  const modeToggle = document.getElementById('modeToggle');
  const statusPill = document.getElementById('statusPill');
  const modelSummaryPill = document.getElementById('modelSummaryPill');

  const QWEN_IMAGE_MODELS = [
    { id: 'qwen-image', name: 'Qwen Image', capability: 'image' },
    { id: 'qwen-image-plus', name: 'Qwen Image Plus', capability: 'image' },
    { id: 'qwen-image-turbo', name: 'Qwen Image Turbo', capability: 'image' },
  ];

  let conversations = [];
  let activeConversationId = null;
  let mode = 'generate';
  let models = [];
  let referenceImages = [];
  let sending = false;

  function text(message, fallback) {
    return typeof t === 'function' ? (t(message) === message ? fallback : t(message)) : fallback;
  }

  function toast(message, type = 'info') {
    if (typeof showToast === 'function') showToast(message, type);
  }

  function createId() {
    if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
    return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }

  function escapeHtml(value) {
    return String(value ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function timeLabel(value) {
    try {
      return new Date(value).toLocaleString();
    } catch {
      return value || '-';
    }
  }

  async function ensureAccess() {
    const stored = await webuiKey.get();
    if (stored && await verifyKey(VERIFY_ENDPOINT, stored)) return true;
    if (stored) webuiKey.clear();
    if (await verifyKey(VERIFY_ENDPOINT, '')) return true;
    location.href = '/webui/login';
    return false;
  }

  async function authHeaders(extra = {}) {
    const key = await webuiKey.get();
    return key ? { ...extra, Authorization: `Bearer ${key}` } : { ...extra };
  }

  async function fetchJson(url, options = {}) {
    const headers = await authHeaders(options.headers || {});
    const response = await fetch(url, { ...options, headers });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload?.detail?.error || payload?.detail?.message || payload?.detail || payload?.error?.message || payload?.error || response.statusText;
      throw new Error(String(message || 'request failed'));
    }
    return payload;
  }

  function currentConversation() {
    return conversations.find((item) => item.id === activeConversationId) || null;
  }

  function persist() {
    const slim = conversations.slice(0, MAX_HISTORY);
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ conversations: slim, activeConversationId }));
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      conversations = Array.isArray(parsed?.conversations) ? parsed.conversations : [];
      activeConversationId = parsed?.activeConversationId || conversations[0]?.id || null;
    } catch {}
  }

  function setStatus(value) {
    statusPill.textContent = value;
  }

  function sortConversations() {
    conversations.sort((a, b) => String(b.updatedAt || '').localeCompare(String(a.updatedAt || '')));
  }

  function newConversation() {
    activeConversationId = null;
    promptInput.value = '';
    referenceImages = [];
    mode = 'generate';
    syncModeButtons();
    renderReferenceBar();
    renderSessions();
    renderFeed();
    promptInput.focus();
  }

  function modelCapability(modelId) {
    const item = models.find((entry) => entry.id === modelId);
    return item?.capability || 'image';
  }

  function modelOptionsForMode(nextMode) {
    return models.filter((item) => {
      if (nextMode === 'edit') {
        return item.capability === 'image_edit' || item.id.startsWith('gpt-image-');
      }
      return item.capability === 'image' || item.id.startsWith('gpt-image-');
    });
  }

  function syncModelOptions() {
    const options = modelOptionsForMode(mode);
    const previous = modelSelect.value;
    modelSelect.innerHTML = '';
    options.forEach((item) => {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.name || item.id;
      modelSelect.appendChild(option);
    });
    if (options.length) {
      modelSelect.value = options.some((item) => item.id === previous) ? previous : options[0].id;
    }
    modelSummaryPill.textContent = `models: ${options.length}`;
  }

  function syncModeButtons() {
    modeToggle.querySelectorAll('button').forEach((button) => {
      button.classList.toggle('active', button.dataset.mode === mode);
    });
    uploadBtn.style.display = mode === 'edit' ? '' : 'none';
    syncModelOptions();
  }

  function selectedCount() {
    let value = Number(countSelect.value || 1);
    if (modelSelect.value.startsWith('gpt-image-')) value = Math.min(value, 4);
    if (mode === 'edit' && modelCapability(modelSelect.value) === 'image_edit') value = Math.min(value, 2);
    return Math.max(1, value);
  }

  function renderReferenceBar() {
    referenceBar.innerHTML = '';
    referenceBar.style.display = referenceImages.length ? 'flex' : 'none';
    referenceImages.forEach((item, index) => {
      const node = document.createElement('div');
      node.className = 'studio-refchip';
      node.innerHTML = `<img src="${item.dataUrl}" alt="ref-${index}"><button type="button" class="studio-refchip-remove">x</button>`;
      node.querySelector('button').addEventListener('click', () => {
        referenceImages.splice(index, 1);
        renderReferenceBar();
      });
      referenceBar.appendChild(node);
    });
  }

  function renderSessions() {
    sessionList.innerHTML = '';
    if (!conversations.length) {
      sessionList.innerHTML = '<div style="padding:10px 12px;color:#8d7f73;font-size:12px;">暂无会话</div>';
      return;
    }
    conversations.forEach((item) => {
      const node = document.createElement('button');
      node.type = 'button';
      node.className = `studio-session-item${item.id === activeConversationId ? ' active' : ''}`;
      node.innerHTML = `
        <div class="studio-session-title">${escapeHtml(item.title || 'Untitled')}</div>
        <div class="studio-session-meta"><span>${escapeHtml(item.mode === 'edit' ? '图生图' : '文生图')}</span><span>${escapeHtml(timeLabel(item.updatedAt))}</span></div>
      `;
      node.addEventListener('click', () => {
        activeConversationId = item.id;
        renderSessions();
        renderFeed();
      });
      sessionList.appendChild(node);
    });
  }

  function renderImages(images) {
    return images.map((item) => {
      if (item.error) {
        return `<div class="studio-image-card error">${escapeHtml(item.error)}</div>`;
      }
      return `<a class="studio-image-card" href="${item.src}" target="_blank" rel="noreferrer"><img src="${item.src}" alt="generated image"></a>`;
    }).join('');
  }

  function renderFeed() {
    const convo = currentConversation();
    if (!convo || !Array.isArray(convo.turns) || !convo.turns.length) {
      emptyState.style.display = '';
      return;
    }
    emptyState.style.display = 'none';
    feed.innerHTML = convo.turns.map((turn) => {
      const refs = Array.isArray(turn.references) && turn.references.length
        ? `<div class="studio-ref-strip">${turn.references.map((item) => `<div class="studio-ref"><img src="${item.dataUrl}" alt="reference"></div>`).join('')}</div>`
        : '';
      return `
        <section class="studio-turn">
          <div class="studio-turn-head">
            <div class="studio-turn-prompt">${escapeHtml(turn.prompt)}</div>
            <div class="studio-turn-tags">
              <span class="studio-tag">${escapeHtml(turn.mode === 'edit' ? '图生图' : '文生图')}</span>
              <span class="studio-tag">${escapeHtml(turn.model)}</span>
              <span class="studio-tag">${escapeHtml(String(turn.count || 1))} 张</span>
            </div>
          </div>
          ${refs}
          <div class="studio-grid">${renderImages(turn.images || [])}</div>
        </section>
      `;
    }).join('');
  }

  async function loadModels() {
    const payload = await fetchJson(MODELS_ENDPOINT);
    const data = Array.isArray(payload?.data) ? payload.data : [];
    const seen = new Set();
    models = [];
    data.forEach((item) => {
      const id = String(item?.id || '').trim();
      if (!id || seen.has(id)) return;
      seen.add(id);
      models.push({ id, name: String(item?.name || id), capability: String(item?.capability || 'chat') });
    });
    if (models.some((item) => item.id.startsWith('qwen')) ) {
      QWEN_IMAGE_MODELS.forEach((item) => {
        if (seen.has(item.id)) return;
        seen.add(item.id);
        models.push(item);
      });
    }
    syncModelOptions();
  }

  async function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result || ''));
      reader.onerror = () => reject(new Error('读取参考图失败'));
      reader.readAsDataURL(file);
    });
  }

  async function handleUpload(files) {
    const valid = Array.from(files || []).filter((file) => file.type.startsWith('image/'));
    if (!valid.length) return;
    const next = await Promise.all(valid.map(async (file) => ({ name: file.name, dataUrl: await fileToDataUrl(file) })));
    referenceImages.push(...next);
    mode = 'edit';
    syncModeButtons();
    renderReferenceBar();
  }

  function normalizeImageItems(data) {
    return (Array.isArray(data) ? data : []).map((item, index) => {
      if (item?.b64_json) {
        return { id: `${Date.now()}-${index}`, src: `data:image/png;base64,${item.b64_json}` };
      }
      if (item?.url) {
        return { id: `${Date.now()}-${index}`, src: String(item.url) };
      }
      return { id: `${Date.now()}-${index}`, error: '图片返回为空' };
    });
  }

  async function submit() {
    const prompt = String(promptInput.value || '').trim();
    if (!prompt) {
      toast('请输入提示词', 'error');
      return;
    }
    if (mode === 'edit' && !referenceImages.length) {
      toast('请先上传至少一张参考图', 'error');
      return;
    }
    if (!modelSelect.value) {
      toast('当前没有可用模型', 'error');
      return;
    }

    sending = true;
    sendBtn.disabled = true;
    setStatus('running');
    try {
      let payload;
      if (mode === 'edit') {
        const form = new FormData();
        form.append('model', modelSelect.value);
        form.append('prompt', prompt);
        form.append('n', String(selectedCount()));
        form.append('response_format', 'b64_json');
        for (const item of referenceImages) {
          const response = await fetch(item.dataUrl);
          const blob = await response.blob();
          form.append('image', new File([blob], item.name || 'reference.png', { type: blob.type || 'image/png' }));
        }
        payload = await fetchJson(EDIT_ENDPOINT, { method: 'POST', body: form, headers: {} });
      } else {
        payload = await fetchJson(GENERATE_ENDPOINT, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: modelSelect.value,
            prompt,
            n: selectedCount(),
            response_format: 'b64_json',
          }),
        });
      }

      const images = normalizeImageItems(payload?.data);
      const now = new Date().toISOString();
      const turn = {
        id: createId(),
        prompt,
        model: modelSelect.value,
        mode,
        count: selectedCount(),
        references: referenceImages.map((item) => ({ name: item.name, dataUrl: item.dataUrl })),
        images,
        createdAt: now,
      };
      let convo = currentConversation();
      if (!convo) {
        convo = {
          id: createId(),
          title: prompt.length > 18 ? `${prompt.slice(0, 18)}...` : prompt,
          mode,
          turns: [],
          updatedAt: now,
        };
        conversations.unshift(convo);
        activeConversationId = convo.id;
      }
      convo.turns = [...(convo.turns || []), turn];
      convo.updatedAt = now;
      convo.mode = mode;
      sortConversations();
      conversations = conversations.slice(0, MAX_HISTORY);
      persist();
      renderSessions();
      renderFeed();
      promptInput.value = '';
      referenceImages = [];
      mode = 'generate';
      syncModeButtons();
      renderReferenceBar();
      setStatus('completed');
    } catch (error) {
      toast(error instanceof Error ? error.message : '图片请求失败', 'error');
      setStatus('failed');
    } finally {
      sending = false;
      sendBtn.disabled = false;
      promptInput.focus();
    }
  }

  function bindEvents() {
    newConversationBtn.addEventListener('click', newConversation);
    sendBtn.addEventListener('click', () => { if (!sending) void submit(); });
    uploadBtn.addEventListener('click', () => referenceInput.click());
    referenceInput.addEventListener('change', () => { void handleUpload(referenceInput.files); referenceInput.value = ''; });
    promptInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        if (!sending) void submit();
      }
    });
    modeToggle.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        mode = button.dataset.mode || 'generate';
        syncModeButtons();
      });
    });
    modelSelect.addEventListener('change', () => {
      const max = modelSelect.value.startsWith('gpt-image-') ? 4 : modelCapability(modelSelect.value) === 'image_edit' ? 2 : 4;
      if (Number(countSelect.value) > max) countSelect.value = String(max);
    });
  }

  (async () => {
    if (!await ensureAccess()) return;
    await renderWebuiHeader?.();
    await renderSiteFooter?.();
    loadState();
    bindEvents();
    await loadModels().catch((error) => toast(error.message, 'error'));
    renderReferenceBar();
    renderSessions();
    renderFeed();
    setStatus('ready');
  })();
})();
