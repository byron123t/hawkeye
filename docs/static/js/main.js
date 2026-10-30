/* Hawkeye project page: demo clip switcher, speed/cost section, BibTeX copy, theme toggle. */
(function () {
  "use strict";

  /* ---------------- demo video ---------------- */
  var video = document.getElementById("demo-video");
  var source = document.getElementById("demo-src");
  var missing = document.getElementById("player-missing");
  var buttons = document.querySelectorAll(".seg-btn");

  function showMissing(on) {
    if (!missing) return;
    missing.hidden = !on;
    if (video) video.style.visibility = on ? "hidden" : "visible";
  }
  function load(clip) {
    if (!video || !source) return;
    showMissing(false);
    video.poster = "static/videos/" + clip + ".jpg";
    source.src = "static/videos/" + clip + ".mp4";
    video.load();
    var p = video.play();
    if (p && p.catch) p.catch(function () { /* autoplay blocked: poster stays */ });
  }
  if (source) source.addEventListener("error", function () { showMissing(true); });
  if (video) video.addEventListener("error", function () { showMissing(true); });
  buttons.forEach(function (b) {
    b.addEventListener("click", function () {
      buttons.forEach(function (o) { o.classList.remove("is-active"); o.setAttribute("aria-selected", "false"); });
      b.classList.add("is-active"); b.setAttribute("aria-selected", "true");
      load(b.getAttribute("data-clip"));
    });
  });

  /* ---------------- speed & cost ---------------- */
  var COLORS = { h4: "var(--c-h4)", h8: "var(--c-h8)", base: "var(--c-base)", api: "var(--c-api)" };
  function colorOf(m) {
    var n = (m.name || "").toLowerCase();
    if (n.indexOf("hawkeye") >= 0 && n.indexOf("8b") >= 0) return COLORS.h8;
    if (n.indexOf("hawkeye") >= 0) return COLORS.h4;
    if (m.kind === "local") return COLORS.base;
    return COLORS.api;
  }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function fmt(x, d) { return (x === null || x === undefined || isNaN(x)) ? "&ndash;" : Number(x).toFixed(d); }
  function money(x) {
    if (x === null || x === undefined || isNaN(x)) return "&ndash;";
    if (x === 0) return "$0";
    return "$" + (x < 10 ? Number(x).toFixed(2) : Number(x).toFixed(0));
  }

  function gpuPrice(data) {
    var el = document.getElementById("gpu-price");
    var v = el ? parseFloat(el.value) : NaN;
    return isNaN(v) || v < 0 ? (data.gpu_price_usd_per_h || 0) : v;
  }

  function num(id, dflt) {
    var el = document.getElementById(id);
    var v = el ? parseFloat(el.value) : NaN;
    return isNaN(v) || v < 0 ? dflt : v;
  }

  function dailyCost(m, data) {
    var c = costOf(m, data);
    if (c === null || c === undefined || isNaN(c)) return null;
    return c / 1000 * num("reads-per-s", 1) * 3600 * num("hours-per-day", 16);
  }

  function costOf(m, data) {
    if (m.kind === "local" && m.gpu_hours_per_1k != null) return m.gpu_hours_per_1k * gpuPrice(data);
    return m.cost_per_1k_frames_usd;
  }

  function renderTable(data) {
    var tb = document.querySelector("#speed-table tbody");
    if (!tb) return;
    var rows = data.models.slice().sort(function (a, b) { return a.latency_s - b.latency_s; });
    tb.innerHTML = rows.map(function (m) {
      var ours = (m.name || "").toLowerCase().indexOf("hawkeye") >= 0;
      return '<tr class="' + (ours ? "is-ours" : "") + '">' +
        '<td><span class="swatch" style="background:' + colorOf(m) + '"></span><span class="mname">' + esc(m.name) + "</span>" +
        ' <span class="pill">' + (m.kind === "local" ? "local" : "API") + "</span></td>" +
        "<td>" + fmt(m.bstrb_video_f1, 3) + "</td>" +
        "<td>" + fmt(m.vram_gb, 1) + "</td>" +
        "<td>" + fmt(m.tflops_per_crop, 1) + "</td>" +
        "<td>" + fmt(m.ttft_s, 2) + "</td>" +
        "<td>" + (m.decode_tok_s == null ? "&mdash;" : fmt(m.decode_tok_s, 0)) + "</td>" +
        "<td>" + fmt(m.latency_s, 2) + "</td>" +
        "<td>" + money(costOf(m, data)) + (m.kind === "local" ? "*" : "") + "</td>" +
        "<td>" + money(dailyCost(m, data)) + "</td></tr>";
    }).join("");
  }

  function renderChart(data) {
    var host = document.getElementById("speed-chart");
    if (!host) return;
    var rows = data.models.slice().sort(function (a, b) { return a.latency_s - b.latency_s; });
    var avail = host.clientWidth || 760;
    var W = Math.max(300, Math.min(760, avail));          // draw at the real width: text stays legible
    var narrow = W < 520;
    var labelW = narrow ? 118 : 190, valW = narrow ? 46 : 60, rowH = narrow ? 28 : 30, barH = 14, top = 26, bottom = 26;
    var H = top + rows.length * rowH + bottom;
    var max = Math.max.apply(null, rows.map(function (m) { return m.latency_s || 0; })) || 1;
    var step = max > 20 ? 5 : max > 8 ? 2 : max > 4 ? 1 : 0.5;
    var axisMax = Math.ceil(max / step) * step;
    var plotW = W - labelW - valW;
    var x = function (v) { return labelW + (v / axisMax) * plotW; };
    var s = ['<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" role="img" aria-label="Median latency per crop for each model"' +
             (narrow ? ' class="narrow"' : "") + ">"];
    s.push('<text class="ctitle" x="' + (narrow ? 0 : labelW) + '" y="14">Median latency per crop (seconds)</text>');
    for (var t = 0; t <= axisMax + 1e-9; t += step) {
      var gx = x(t).toFixed(1);
      s.push('<line class="gridline" x1="' + gx + '" x2="' + gx + '" y1="' + (top - 4) + '" y2="' + (H - bottom + 4) + '"/>');
      s.push('<text class="axis" x="' + gx + '" y="' + (H - 6) + '" text-anchor="middle">' + (Math.round(t * 10) / 10) + "</text>");
    }
    rows.forEach(function (m, i) {
      var y = top + i * rowH + (rowH - barH) / 2, w = Math.max(2, x(m.latency_s) - labelW);
      s.push('<text class="lbl" x="' + (labelW - 10) + '" y="' + (y + barH - 2) + '" text-anchor="end">' + esc(m.name) + "</text>");
      s.push('<rect x="' + labelW + '" y="' + y + '" width="' + w.toFixed(1) + '" height="' + barH + '" rx="4" style="fill:' + colorOf(m) + '"/>');
      s.push('<text class="val" x="' + (labelW + w + 6).toFixed(1) + '" y="' + (y + barH - 2) + '">' + fmt(m.latency_s, 1) + " s</text>");
    });
    s.push("</svg>");
    host.innerHTML = s.join("");
  }

  function renderSpeed(data) {
    if (!data || !data.models || !data.models.length) return;
    renderTable(data);
    renderChart(data);
    var pe = document.getElementById("gpu-price");
    if (pe) {
      if (!pe.value) pe.value = (data.gpu_price_usd_per_h || 0).toFixed(2);
      pe.addEventListener("input", function () { renderTable(data); });
    }
    ["reads-per-s", "hours-per-day"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.addEventListener("input", function () { renderTable(data); });
    });
    var t = null;
    window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { renderChart(data); }, 150); });
    var note = document.getElementById("speed-note");
    if (note) {
      var bits = [];
      if (data.frames) bits.push(data.frames + " crops from the demonstration recordings");
      if (data.gpu) bits.push("local models on one " + data.gpu);
      var txt = bits.join("; ");
      if (data.note) txt += (txt ? ". " : "") + data.note;
      note.textContent = txt;
    }
  }

  function loadSpeed() {
    var inline = document.getElementById("speed-data");
    var raw = inline ? inline.textContent.trim() : "";
    if (raw) {
      try { renderSpeed(JSON.parse(raw)); return; } catch (e) { /* fall through to fetch */ }
    }
    if (!window.fetch) return;
    fetch("static/data/speed.json", { cache: "no-cache" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(renderSpeed)
      .catch(function () { /* keep the placeholder row */ });
  }
  loadSpeed();

  /* ---------------- BibTeX copy ---------------- */
  var copy = document.getElementById("copy-bib");
  if (copy) copy.addEventListener("click", function () {
    var text = document.querySelector("#bib code").innerText, btn = this;
    var done = function () {
      btn.innerHTML = '<i class="fa-solid fa-check"></i> Copied';
      setTimeout(function () { btn.innerHTML = '<i class="fa-regular fa-copy"></i> Copy'; }, 1600);
    };
    if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {});
  });

  /* ---------------- theme toggle ---------------- */
  var root = document.documentElement, key = "hawkeye-theme";
  try { var saved = localStorage.getItem(key); if (saved) root.setAttribute("data-theme", saved); } catch (e) {}
  var tog = document.getElementById("theme-toggle");
  if (tog) tog.addEventListener("click", function () {
    var cur = root.getAttribute("data-theme");
    var dark = cur ? cur === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem(key, next); } catch (e) {}
  });
})();

/* Join the last two words of each paragraph with a non-breaking space so that no paragraph ends
   with a single word on its own line (for browsers without text-wrap: pretty). */
(function () {
  var sel = "p, figcaption, li, .lbl";
  document.querySelectorAll(sel).forEach(function (el) {
    if (el.closest("pre, code, table")) return;
    var walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    var last = null, n;
    while ((n = walker.nextNode())) { if (n.nodeValue.trim()) last = n; }
    if (!last) return;
    var v = last.nodeValue.replace(/\s+$/, "");
    var i = v.lastIndexOf(" ");
    if (i > 0 && v.length - i < 14) last.nodeValue = v.slice(0, i) + "\u00a0" + v.slice(i + 1);
  });
})();

