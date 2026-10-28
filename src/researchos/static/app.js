"use strict";

const $ = (selector) => document.querySelector(selector);
let current = null;
let activeTab = "plan";
let listed = [];
let selectionVersion = 0;
let listVersion = 0;
let embeddingsExpected = false;
let modelConfigured = false;
let passageQuery = "";
let passageResult = null;

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function notify(message = "") {
  $("#notice").textContent = message;
  $("#notice").hidden = !message;
}

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, signal: AbortSignal.timeout(15000) });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    const detail = typeof body.detail === "string" ? body.detail : "Please check your input and try again.";
    throw new Error(response.status >= 500 ? "The server could not complete this request. Please try again." : detail);
  }
  return response;
}

async function action(button, callback) {
  button.disabled = true;
  notify();
  try { await callback(); }
  catch (error) {
    notify(error.name === "TimeoutError" ? "The request timed out. Refresh the session list before retrying." : error.message);
  } finally { button.disabled = false; }
}

const label = (value) => value.replaceAll("_", " ").toLowerCase().replace(/^./, (c) => c.toUpperCase());
const date = (value) => new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });

function renderSidebar() {
  const nav = $("#sessions");
  nav.replaceChildren();
  if (!listed.length) nav.append(element("p", "small muted", "Your saved research will appear here."));
  for (const session of listed) {
    const button = element("button", `session-link${current?.id === session.id ? " active" : ""}`);
    button.title = session.question;
    if (current?.id === session.id) button.setAttribute("aria-current", "page");
    button.append(element("strong", "", session.question), element("small", "", date(session.updated_at)));
    button.addEventListener("click", () => action(button, () => openSession(session.id)));
    nav.append(button);
  }
}

async function loadSessions(append = false) {
  const version = ++listVersion;
  const sessions = await (await api(`/research?limit=30&offset=${append ? listed.length : 0}`)).json();
  if (version !== listVersion) return;
  listed = append ? [...listed, ...sessions.filter((s) => !listed.some((old) => old.id === s.id))] : sessions;
  $("#load-more").hidden = sessions.length < 30;
  renderSidebar();
}

async function openSession(id) {
  const version = ++selectionVersion;
  const session = await (await api(`/research/${encodeURIComponent(id)}`)).json();
  if (version !== selectionVersion) return;
  current = session;
  history.replaceState(null, "", `#${session.id}`);
  renderSession();
  $("#session-title").focus();
}

function newSession() {
  ++selectionVersion;
  current = null;
  history.replaceState(null, "", location.pathname);
  $("#welcome").hidden = false;
  $("#workspace").hidden = true;
  notify();
  renderSidebar();
  $("#question").focus();
}

function empty(title, description) {
  const node = element("div", "empty");
  node.append(element("div", "empty-icon", "◎"), element("h3", "", title), element("p", "", description));
  return node;
}

function sourceLink(source) {
  const url = new URL(source.url);
  if (!["http:", "https:"].includes(url.protocol)) return element("span", "", source.title);
  const link = element("a", "", `${source.title} ↗`);
  link.href = url.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

const BUSY = ["searching", "reading", "analyzing"];

function readState(source) {
  const events = (current.read_history || []).filter((event) => event.source_id === source.id);
  const done = events.findLast((event) => event.status !== "failed");
  return {
    document: done && current.documents.find((doc) => doc.id === done.document_id),
    duplicate: done?.status === "duplicate",
    error: done ? null : events.at(-1)?.error,
  };
}

function documentViewer(doc) {
  const details = element("details", "document");
  details.append(element("summary", "", "View extracted text"));
  details.addEventListener("toggle", async () => {
    if (!details.open || details.dataset.loaded) return;
    details.dataset.loaded = "true";
    const status = element("p", "small muted", "Loading…");
    details.append(status);
    try {
      const { sections } = await (await api(`/research/${current.id}/documents/${doc.id}`)).json();
      status.remove();
      for (const section of sections) {
        details.append(element("h4", "", `${section.heading || "Untitled section"}${section.page ? ` · Page ${section.page}` : ""}`), element("p", "document-text", section.text));
      }
    } catch (error) {
      delete details.dataset.loaded;
      status.textContent = `Could not load the text: ${error.message}`;
    }
  });
  return details;
}

function renderSession() {
  $("#welcome").hidden = true;
  $("#workspace").hidden = false;
  $("#session-title").textContent = current.question;
  $("#session-date").textContent = `Created ${date(current.created_at)}`;
  $("#session-status").textContent = current.status === "awaiting_providers" ? "Plan ready · Discovery pending" : label(current.status);
  $("#status-description").textContent = current.status === "awaiting_providers"
    ? "Your plan is saved. Continue discovery to search configured providers."
    : (current.events.at(-1)?.message || "Session saved.");
  $("#metrics").replaceChildren();
  for (const [title, value] of [["Research tasks", current.tasks.length], ["Sources found", current.sources.length], ["Documents read", current.documents.length], ["Claims extracted", current.claims.length], ["Search attempts", (current.search_history || []).length], ["Model tokens", current.token_usage]]) {
    const metric = element("div", "metric");
    metric.append(element("span", "", title), element("strong", "", value.toLocaleString()));
    $("#metrics").append(metric);
  }
  $("#continue").hidden = !["awaiting_providers", "discovery_failed", "interrupted"].includes(current.status);
  const unread = current.sources.some((source) => !readState(source).document) && current.documents.length < current.budget.max_documents;
  const unindexed = current.documents.some((doc) => !doc.chunk_count || (embeddingsExpected && !doc.embedding_model));
  $("#read").hidden = !(["discovery_complete", "reading_complete", "reading_failed"].includes(current.status) && (unread || unindexed));
  $("#read").textContent = current.status === "discovery_complete" ? "Read sources" : (unread ? "Retry unread sources" : "Index documents");
  const unanalyzed = current.tasks.some((task) => ["pending", "failed"].includes(task.analysis || "pending"));
  $("#analyze").hidden = !(modelConfigured && ["reading_complete", "analysis_complete", "analysis_failed"].includes(current.status) && unanalyzed && current.documents.some((doc) => doc.chunk_count));
  $("#analyze").textContent = current.claims.length ? "Analyze remaining tasks" : "Analyze evidence";
  renderPanel();
  renderSidebar();
}

function renderPanel() {
  const panel = $("#panel");
  panel.replaceChildren();
  panel.setAttribute("aria-labelledby", `tab-${activeTab}`);
  for (const tab of document.querySelectorAll("[data-tab]")) {
    tab.setAttribute("aria-selected", String(tab.dataset.tab === activeTab));
    tab.tabIndex = tab.dataset.tab === activeTab ? 0 : -1;
  }
  if (activeTab === "plan") {
    panel.append(element("p", "small muted", `${current.plan_source === "model" ? "A plan written by the model for this question." : "A template investigation plan."} Task status reflects source discovery only; analysis is not implemented yet.`));
    current.tasks.forEach((task, index) => {
      const card = element("article", "task");
      const content = element("div", "task-content");
      content.append(element("h3", "", task.question));
      task.source_types.forEach((type) => content.append(element("span", "tag", label(type))));
      if (task.query) content.append(element("p", "small muted", `Search: ${task.query}`));
      card.append(element("span", "task-number", String(index + 1).padStart(2, "0")), content, element("span", "tag", label(task.status)));
      panel.append(card);
    });
  } else if (activeTab === "sources") {
    if (!current.sources.length) panel.append(empty("No sources discovered yet", "No results have been saved. Check the timeline for search progress, provider failures, or configuration details."));
    for (const source of current.sources) {
      const card = element("article", "source");
      card.append(sourceLink(source), element("p", "muted small", `${label(source.source_type)} · ${(source.providers?.length ? source.providers : [source.provider]).join(", ")} · Retrieved ${date(source.retrieved_at)}`));
      if (source.authors?.length) card.append(element("p", "small muted", source.authors.join(", ")));
      if (source.published_at) card.append(element("p", "small muted", `Published: ${source.published_at}`));
      if (source.snippet) card.append(element("p", "", source.snippet));
      card.append(element("span", "tag", "Discovery metadata · Not verified evidence"));
      const read = readState(source);
      if (read.document) {
        const doc = read.document;
        card.append(element("span", "tag read", `${read.duplicate ? "Same content as another source" : "Read"} · ${doc.section_count} sections${doc.pages ? ` · ${doc.pages} pages` : ""} · ${doc.content_type.toUpperCase()}${doc.chunk_count ? ` · ${doc.chunk_count} passages${doc.embedding_model ? "" : " (keyword only)"}` : ""}${doc.truncated ? " · Truncated" : ""}`), documentViewer(doc));
      } else if (read.error) {
        card.append(element("p", "small unread", `Not read: ${read.error}`));
      }
      panel.append(card);
    }
  } else if (activeTab === "passages") {
    const indexed = current.documents.reduce((total, doc) => total + (doc.chunk_count || 0), 0);
    if (!indexed) {
      panel.append(empty("No passages indexed yet", "Read the discovered sources first. Each document is split into passages that can be searched by meaning and by keyword."));
    } else {
      panel.append(element("p", "small muted", `${indexed.toLocaleString()} passages from ${current.documents.length} documents${current.embedding_model ? ` · embedded with ${current.embedding_model}` : " · keyword search only"}. Passages are source text, not verified claims.`));
      const form = element("form", "passage-search");
      const input = element("input");
      input.type = "search";
      input.value = passageQuery;
      input.placeholder = "Search the documents this session has read";
      input.setAttribute("aria-label", "Search passages");
      input.minLength = 2;
      input.maxLength = 500;
      input.required = true;
      const submit = element("button", "primary", "Search");
      submit.type = "submit";
      form.append(input, submit);
      form.addEventListener("submit", (event) => {
        event.preventDefault();
        action(submit, async () => {
          const id = current.id;
          passageQuery = input.value.trim();
          const result = await (await api(`/research/${id}/search?q=${encodeURIComponent(passageQuery)}&limit=8`)).json();
          if (current?.id === id) { passageResult = { id, ...result }; renderPanel(); }
        });
      });
      panel.append(form);
      if (passageResult?.id === current.id) {
        panel.append(element("p", "small muted", passageResult.mode === "hybrid" ? "Ranked by meaning and keywords combined." : "Ranked by keywords only; the embedding model was unavailable."));
        if (!passageResult.results.length) panel.append(element("p", "muted", "No matching passages."));
        for (const item of passageResult.results) {
          const card = element("article", "claim");
          const source = current.sources.find((entry) => entry.id === item.source_id);
          card.append(source ? sourceLink(source) : element("strong", "", item.source_title), element("p", "small muted", `${item.chunk.heading || "Untitled section"}${item.chunk.page ? ` · Page ${item.chunk.page}` : ""}`), element("blockquote", "", item.chunk.text));
          panel.append(card);
        }
      }
    }
  } else if (activeTab === "evidence") {
    if (!current.claims.length) panel.append(empty("Evidence comes before conclusions", "Once documents are read and analyzed, atomic claims and the exact passages they quote will appear here. No claims have been generated yet."));
    else panel.append(element("p", "small muted", "Each claim quotes the passage it cites, and the quote was checked against the stored text. The verdict is the model's second-pass check of the claim against that passage, not independent confirmation."));
    for (const claim of current.claims) {
      const card = element("article", "claim");
      const task = current.tasks.find((entry) => entry.id === claim.task_id);
      if (task) card.append(element("p", "small muted", task.question));
      card.append(element("strong", "", claim.text), element("p", "", ""), element("span", `tag verdict ${claim.verification.toLowerCase()}`, label(claim.verification)), element("span", "small muted", claim.verification_reason));
      const details = element("details");
      details.append(element("summary", "", "Show evidence"));
      const evidence = current.evidence.filter((item) => item.claim_id === claim.id && claim.evidence_ids.includes(item.id));
      if (!evidence.length) details.append(element("p", "muted", "No linked evidence is available."));
      for (const item of evidence) {
        const document = current.documents.find((doc) => doc.id === item.document_id);
        const source = current.sources.find((entry) => entry.id === document?.source_id);
        details.append(element("blockquote", "", item.text));
        if (source) details.append(sourceLink(source));
        details.append(element("p", "small muted", `${item.heading ? `${item.heading} · ` : ""}${item.page ? `Page ${item.page} · ` : ""}${label(item.evidence_type)} evidence · Extracted ${date(item.extracted_at)}`));
      }
      card.append(details);
      panel.append(card);
    }
  } else if (activeTab === "timeline") {
    panel.append(element("p", "small muted", "Saved activity · Updates automatically while work is in progress."));
    for (const event of current.events) {
      const entry = element("article", "timeline-entry");
      entry.append(element("strong", "", label(event.action)), element("p", "", event.message), element("time", "", date(event.timestamp)));
      panel.append(entry);
    }
    for (const error of current.errors) panel.append(element("p", "danger", error));
  } else if (activeTab === "report") {
    panel.append(current.report ? element("div", "report", current.report) : empty("Your report will take shape here", "A cited report requires discovered sources, extracted evidence, and verified claims. This session has not reached that stage yet."));
  }
}

$("#research-form").addEventListener("submit", (event) => {
  event.preventDefault();
  action($("#create"), async () => {
    const data = new FormData(event.target);
    const question = data.get("question").trim();
    if (question.length < 3) throw new Error("Please enter a research question of at least 3 characters.");
    const budget = Object.fromEntries(["max_searches", "max_sources", "max_tokens"].map((key) => [key, Number(data.get(key))]));
    const version = ++selectionVersion;
    const session = await (await api("/research", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, budget }) })).json();
    if (version === selectionVersion) {
      current = session;
      activeTab = "plan";
      history.replaceState(null, "", `#${session.id}`);
      renderSession();
      $("#session-title").focus();
    }
    await loadSessions();
  });
});

$("#new-session").addEventListener("click", newSession);
$("#refresh-list").addEventListener("click", (event) => action(event.currentTarget, () => loadSessions()));
$("#load-more").addEventListener("click", (event) => action(event.currentTarget, () => loadSessions(true)));
$("#refresh-session").addEventListener("click", (event) => action(event.currentTarget, () => openSession(current.id)));
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => {
  $("#question").value = button.dataset.question;
  $("#question").focus();
}));
const tabs = [...document.querySelectorAll("[data-tab]")];
tabs.forEach((button, index) => {
  button.addEventListener("click", () => { activeTab = button.dataset.tab; renderPanel(); });
  button.addEventListener("keydown", (event) => {
    const target = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 }[event.key];
    if (target === undefined) return;
    event.preventDefault();
    tabs[target].click();
    tabs[target].focus();
  });
});
$("#export").addEventListener("click", (event) => action(event.currentTarget, async () => {
  const id = current.id;
  const blob = await (await api(`/research/${id}/export`, { method: "POST" })).blob();
  const url = URL.createObjectURL(blob);
  const link = element("a");
  link.href = url;
  link.download = `research-${id}.json`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}));
$("#delete").addEventListener("click", (event) => {
  const id = current.id;
  if (!confirm("Permanently delete this research session and its saved data?")) return;
  action(event.currentTarget, async () => {
    await api(`/research/${id}`, { method: "DELETE" });
    if (current?.id === id) newSession();
    await loadSessions();
  });
});

async function initialize() {
  try {
    const health = await (await api("/health")).json();
    embeddingsExpected = (health.embedding_models || []).length > 0;
    modelConfigured = Boolean(health.model);
    await loadSessions();
    if (location.hash.slice(1)) await openSession(location.hash.slice(1));
  } catch (error) { notify(`Unable to load the workspace: ${error.message}`); }
}
$("#continue").addEventListener("click", (event) => action(event.currentTarget, async () => {
  const id = current.id;
  const session = await (await api(`/research/${id}/continue`, {method: "POST"})).json();
  if (current?.id === id) { current = session; renderSession(); }
}));

$("#analyze").addEventListener("click", (event) => action(event.currentTarget, async () => {
  const id = current.id;
  const session = await (await api(`/research/${id}/analyze`, {method: "POST"})).json();
  if (current?.id === id) { current = session; activeTab = "evidence"; renderSession(); }
}));
$("#read").addEventListener("click", (event) => action(event.currentTarget, async () => {
  const id = current.id;
  const session = await (await api(`/research/${id}/read`, {method: "POST"})).json();
  if (current?.id === id) { current = session; renderSession(); }
}));

async function pollSession() {
  try {
    if (BUSY.includes(current?.status)) {
      const id = current.id;
      const version = selectionVersion;
      const session = await (await api(`/research/${id}`)).json();
      if (current?.id === id && version === selectionVersion) {
        current = session;
        renderSession();
        if (!BUSY.includes(session.status)) await loadSessions();
      }
    }
  } catch (error) { notify(`Progress refresh failed: ${error.message}`); }
  finally { setTimeout(pollSession, 2000); }
}
initialize();
pollSession();
