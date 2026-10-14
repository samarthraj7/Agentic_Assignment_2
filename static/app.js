// POST + streamed SSE: native EventSource can't POST, so read the body with fetch().
const $ = (id) => document.getElementById(id);
const logEl = $("log"), traceEl = $("trace"), citesEl = $("citations"), routingEl = $("routing");
let sessionId = localStorage.getItem("a2_session") || newSession();

function newSession() {
  const id = crypto.randomUUID();
  localStorage.setItem("a2_session", id);
  return id;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

const anchor = (id) => "cite-" + id.replace(/[^\w-]/g, "_");

function linkCitations(node) {
  const html = node.textContent
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/\[([\w./-]+\.mdx?#\d+)\]/g, (_, id) => `<a class="cite" href="#${anchor(id)}">[${id}]</a>`);
  node.innerHTML = html;
}

function traceLine(ev) {
  const li = el("li");
  li.append(el("span", `badge ${ev.type}`, ev.type));
  const { type, ...rest } = ev;
  li.append(document.createTextNode(JSON.stringify(rest)));
  traceEl.append(li);
  li.scrollIntoView({ block: "nearest" });
}

function routeCard(ev, turnNo) {
  const d = el("div", "route");
  if (ev.type === "agent_route") {
    d.innerHTML = `<span class="dim">turn ${turnNo} · hop ${ev.hop}</span><br>supervisor → <b></b><br><span class="dim"></span>`;
    d.querySelector("b").textContent = ev.to;
    d.querySelector(".dim:last-child").textContent = ev.reason;
  } else {
    d.innerHTML = `<span class="dim">turn ${turnNo} · ${ev.dimension}</span><br><b></b> → <span></span><br><span class="dim"></span>`;
    d.querySelector("b").textContent = ev.agent;
    d.querySelector("span:not(.dim)").textContent = `${ev.provider} (${ev.model})`;
    d.querySelector(".dim:last-child").textContent = ev.reason;
  }
  routingEl.prepend(d);
}

function addCitation(c) {
  const li = el("li");
  li.id = anchor(c.id);
  const a = el("a", "cite", c.id);
  a.href = c.url; a.target = "_blank"; a.rel = "noopener";
  li.append(a, document.createTextNode(` — ${c.title}${c.section ? " › " + c.section : ""} (score ${c.score})`));
  citesEl.append(li);
}

function parseFrames(buffer, onFrame) {
  const frames = buffer.split(/\r?\n\r?\n/);
  const rest = frames.pop();
  for (const frame of frames) {
    let event = "message", data = "";
    for (const line of frame.split(/\r?\n/)) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
    if (data) onFrame(event, JSON.parse(data));
  }
  return rest;
}

let turnNo = 0;
async function send(query) {
  turnNo += 1;
  const turn = el("div", "turn");
  turn.append(el("div", "user", query));
  logEl.append(turn);
  citesEl.innerHTML = "";
  traceEl.append(el("li", "dim", `— turn ${turnNo} —`));
  let seg = null, segAgent = null;

  const onFrame = (event, data) => {
    if (event === "token") {
      if (!seg || data.agent !== segAgent) {
        if (seg) linkCitations(seg.body);
        segAgent = data.agent;
        seg = el("div", "seg");
        seg.append(el("span", "who", `${data.agent || "agent"} agent`), seg.body = el("div"));
        turn.append(seg);
      }
      seg.body.textContent += data.text;
      logEl.scrollTop = logEl.scrollHeight;
    } else if (event === "trace") {
      traceLine(data);
      if (data.type === "agent_route" || data.type === "llm_route") routeCard(data, turnNo);
    } else if (event === "error") {
      traceLine({ type: "error", ...data });
      if (data.where === "graph") turn.append(el("div", "err", `Error: ${data.detail}`));
    } else if (event === "citation") {
      addCitation(data);
    } else if (event === "done") {
      if (seg) linkCitations(seg.body);
      const who = (data.agents || []).map((a) => `${a.agent}${a.provider ? "@" + a.provider : ""}${a.ok ? "" : " (failed)"}`).join(" → ");
      turn.append(el("div", "meta", `${who} · TTFT ${data.ttft_ms ?? "–"} ms · total ${data.total_ms} ms`));
    }
  };

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ query, session_id: sessionId }),
  });
  if (!res.ok) {
    turn.append(el("div", "err", `HTTP ${res.status}: ${await res.text()}`));
    return;
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer = parseFrames(buffer + value, onFrame);
  }
}

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("query").value.trim();
  if (!q) return;
  $("query").value = "";
  $("send").disabled = true;
  try { await send(q); }
  catch (err) { logEl.append(el("div", "err", `Network error: ${err.message}`)); }
  finally { $("send").disabled = false; $("query").focus(); }
});

$("query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("form").requestSubmit(); }
});

$("reset").addEventListener("click", () => {
  sessionId = newSession();
  turnNo = 0;
  logEl.innerHTML = traceEl.innerHTML = citesEl.innerHTML = routingEl.innerHTML = "";
});
