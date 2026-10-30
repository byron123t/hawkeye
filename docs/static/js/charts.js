/* Interactive figures for the Hawkeye project page: BoW F1 by text height, all-model heatmap,
   change relative to the base model (forest), smart-glasses captures, OCR engines vs. VLMs, and the
   Snellen example chart. Same conventions as the Snellen chart in main.js: SVG drawn at the
   container width, legend, hover and keyboard-focus tooltips with exact values. Every number comes
   from the JSON in static/data/ (inline <script type="application/json"> copies are used first so
   the page also works when opened from disk). */
(function () {
  "use strict";

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function f3(v) { return v == null || isNaN(v) ? "&ndash;" : Number(v).toFixed(3); }
  function pp(v) { return (v >= 0 ? "+" : "&minus;") + Math.abs(v).toFixed(1); }

  function load(id, url, fn) {
    var el = document.getElementById(id), raw = el ? el.textContent.trim() : "";
    var go = function (d) {
      fn(d);
      var t = null;
      window.addEventListener("resize", function () { clearTimeout(t); t = setTimeout(function () { fn(d); }, 150); });
    };
    if (raw) { try { go(JSON.parse(raw)); return; } catch (e) { /* fall through */ } }
    if (!window.fetch) return;
    fetch(url, { cache: "no-cache" }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(go).catch(function () {});
  }

  /* line styles by model, shared with the Snellen chart */
  var STYLE = {
    "Hawkeye 4B": { c: "var(--c-h4)", w: 2.6, d: "", band: true, dot: true, kind: "ours" },
    "Hawkeye 8B": { c: "var(--c-h8)", w: 2.6, d: "", band: true, dot: true, kind: "ours" },
    "Qwen3-VL-4B": { c: "var(--c-base)", w: 2, d: "7 4", kind: "base" },
    "Qwen3-VL-8B": { c: "var(--c-base)", w: 2, d: "2 3", kind: "base" },
    "Qwen3-VL-235B": { c: "var(--c-api)", w: 1.7, d: "9 3 2 3", kind: "api" },
    "Kimi K2.6": { c: "var(--c-api)", w: 1.7, d: "1.5 3", kind: "api" },
    "GPT-6 Astra": { c: "var(--c-api)", w: 1.7, d: "11 4", kind: "api" },
    "GPT-5": { c: "var(--c-api)", w: 1.7, d: "4 2.5", kind: "api" },
    "EasyOCR": { c: "var(--c-eng)", w: 1.9, d: "6 3", kind: "engine" },
    "Tesseract": { c: "var(--c-eng)", w: 1.9, d: "2 2.5", kind: "engine" },
    "Google Vision": { c: "var(--c-eng)", w: 1.9, d: "", kind: "engine" }
  };
  function sty(name) { return STYLE[String(name).replace(" (base)", "")] || { c: "var(--c-api)", w: 1.6, d: "3 3", kind: "api" }; }
  function swatch(name) {
    var s = sty(name);
    return '<svg width="26" height="10" aria-hidden="true"><line x1="1" x2="25" y1="5" y2="5" style="stroke:' + s.c +
           ";stroke-width:" + s.w + (s.d ? ";stroke-dasharray:" + s.d : "") + '"/></svg>';
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo, raw = span / Math.max(1, n), mag = Math.pow(10, Math.floor(Math.log10(raw)));
    var step = [1, 2, 2.5, 5, 10].map(function (k) { return k * mag; }).filter(function (s) { return s >= raw; })[0] || 10 * mag;
    var out = [], v = Math.ceil(lo / step - 1e-9) * step;
    for (; v <= hi + 1e-9; v += step) out.push(Math.round(v * 1e6) / 1e6);
    return out;
  }
  function place(tip, host, x) {
    var w = host.clientWidth, tw = tip.offsetWidth;
    var left = x + 14;
    if (left + tw > w) left = x - tw - 14;
    tip.style.left = Math.max(0, Math.min(left, w - tw)) + "px";
  }
  function bindHits(host, show, hide) {
    host.querySelectorAll(".hit").forEach(function (r) {
      var i = r.getAttribute("data-i");
      r.addEventListener("mouseenter", function () { show(i, r); });
      r.addEventListener("focus", function () { show(i, r); });
      r.addEventListener("mouseleave", hide);
      r.addEventListener("blur", hide);
    });
  }
  function tipRow(name, val, extra) {
    var s = sty(name);
    return '<div class="tip-row"><span class="tip-key" style="border-color:' + s.c + (s.d ? ";border-top-style:dashed" : "") +
           '"></span><span class="tip-n">' + esc(name) + (extra ? ' <span class="tip-ci">' + extra + "</span>" : "") +
           '</span><span class="tip-v">' + val + "</span></div>";
  }

  /* ======================= 1. BoW F1 by text height ======================= */
  var BH = { mode: "base", hidden: {} };
  var BH_SETS = { base: ["Hawkeye 4B", "Hawkeye 8B", "Qwen3-VL-4B", "Qwen3-VL-8B"],
                  api: ["Hawkeye 4B", "Hawkeye 8B", "Qwen3-VL-235B", "Kimi K2.6", "GPT-6 Astra", "GPT-5"] };

  function renderByHeight(data) {
    var host = document.getElementById("bh-chart");
    if (!host || !data || !data.models) return;
    var set = BH_SETS[BH.mode];
    var models = data.models.filter(function (m) { return set.indexOf(m.name) >= 0; });
    var vis = models.filter(function (m) { return !BH.hidden[m.name]; });
    var split = data.split_px || 20;
    var W = Math.max(300, Math.min(1000, host.clientWidth || 800)), narrow = W < 560;
    var H = narrow ? 300 : 330, mt = 26, mb = 44, ml = 44, mr = 44, gap = narrow ? 18 : 30;
    var panels = [
      { lo: 0, hi: split, label: "≤ " + split + " px", w: 0.55 },
      { lo: split, hi: Math.ceil(Math.max.apply(null, data.models[0].points.map(function (p) { return p.x; })) / 5) * 5 + 2, label: "> " + split + " px", w: 0.45 }
    ];
    var pwTot = W - ml - mr - gap, x0 = ml;
    panels.forEach(function (p, k) {
      p.x0 = x0; p.pw = pwTot * p.w; x0 += p.pw + gap;
      var ys = [];
      (vis.length ? vis : models).forEach(function (m) {
        m.points.forEach(function (q) { if ((k === 0 ? q.x <= split : q.x > split)) { ys.push(q.lo, q.hi); } });
      });
      var a = Math.min.apply(null, ys), b = Math.max.apply(null, ys), pad = (b - a) * 0.08 || 0.05;
      p.ylo = Math.max(0, a - pad); p.yhi = Math.min(1, b + pad);
      p.X = function (x) { return p.x0 + (x - p.lo) / (p.hi - p.lo) * p.pw; };
      p.Y = function (v) { return mt + (1 - (v - p.ylo) / (p.yhi - p.ylo)) * (H - mt - mb); };
    });
    var s = ['<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" role="img" aria-label="BoW F1 by mean text height on held-out BSTRB video frames">'];
    panels.forEach(function (p, k) {
      var ticks = niceTicks(p.ylo, p.yhi, 5);
      ticks.forEach(function (v) {
        s.push('<line class="gridline" x1="' + p.x0 + '" x2="' + (p.x0 + p.pw) + '" y1="' + p.Y(v) + '" y2="' + p.Y(v) + '"/>');
        var ax = k === 0 ? p.x0 - 7 : p.x0 + p.pw + 7;
        s.push('<text class="axis" x="' + ax + '" y="' + (p.Y(v) + 4) + '" text-anchor="' + (k === 0 ? "end" : "start") + '">' + v.toFixed(2) + "</text>");
      });
      niceTicks(p.lo, p.hi, k === 0 ? 4 : 3).forEach(function (x) {
        if (x < p.lo || x > p.hi) return;
        s.push('<text class="axis" x="' + p.X(x) + '" y="' + (H - mb + 16) + '" text-anchor="middle">' + x + "</text>");
      });
      s.push('<text class="ctitle" x="' + (p.x0 + 6) + '" y="' + (mt - 8) + '">' + esc(p.label) + "</text>");
      s.push('<rect class="frame" x="' + p.x0 + '" y="' + mt + '" width="' + p.pw + '" height="' + (H - mt - mb) + '"/>');
    });
    s.push('<text class="ctitle" x="' + (ml + pwTot / 2 + gap / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Mean text height (px)</text>');
    s.push('<text class="ctitle" transform="translate(12,' + ((H - mb + mt) / 2) + ') rotate(-90)" text-anchor="middle">BoW F1</text>');
    var order = vis.slice().sort(function (a, b) { return (sty(a.name).band ? 1 : 0) - (sty(b.name).band ? 1 : 0); });
    panels.forEach(function (p, k) {
      var inP = function (q) { return k === 0 ? q.x <= split : q.x > split; };
      order.forEach(function (m) {
        var st = sty(m.name), pts = m.points.filter(inP);
        if (!pts.length) return;
        if (st.band) {
          var up = pts.map(function (q) { return p.X(q.x) + "," + p.Y(q.hi); });
          var dn = pts.map(function (q) { return p.X(q.x) + "," + p.Y(q.lo); }).reverse();
          s.push('<polygon points="' + up.concat(dn).join(" ") + '" style="fill:' + st.c + ';opacity:.13"/>');
        }
      });
      order.forEach(function (m) {
        var st = sty(m.name), pts = m.points.filter(inP);
        if (!pts.length) return;
        s.push('<polyline points="' + pts.map(function (q) { return p.X(q.x).toFixed(1) + "," + p.Y(q.y).toFixed(1); }).join(" ") +
               '" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:' + st.c + ";stroke-width:" + st.w +
               (st.d ? ";stroke-dasharray:" + st.d : "") + '"/>');
        if (st.dot) pts.forEach(function (q) {
          s.push('<circle cx="' + p.X(q.x) + '" cy="' + p.Y(q.y) + '" r="3.6" style="fill:' + st.c + ';stroke:var(--surface);stroke-width:1.8"/>');
        });
      });
    });
    var ref = data.models[0].points;
    s.push('<line id="bh-cross" class="crosshair" x1="0" x2="0" y1="' + mt + '" y2="' + (H - mb) + '" visibility="hidden"/>');
    ref.forEach(function (q, i) {
      var p = q.x <= split ? panels[0] : panels[1];
      var prev = ref[i - 1], next = ref[i + 1];
      var xl = prev && (prev.x <= split) === (q.x <= split) ? (p.X(prev.x) + p.X(q.x)) / 2 : p.x0;
      var xr = next && (next.x <= split) === (q.x <= split) ? (p.X(next.x) + p.X(q.x)) / 2 : p.x0 + p.pw;
      s.push('<rect class="hit" data-i="' + i + '" x="' + xl + '" y="' + mt + '" width="' + Math.max(1, xr - xl) + '" height="' + (H - mt - mb) +
             '" tabindex="0" aria-label="Bin centred at ' + q.x + ' px"/>');
    });
    s.push("</svg>");
    host.innerHTML = s.join("") + '<div class="tip" id="bh-tip" hidden></div>';
    var tip = document.getElementById("bh-tip"), cross = document.getElementById("bh-cross");
    bindHits(host, function (i) {
      i = +i;
      var p = ref[i].x <= split ? panels[0] : panels[1], x = p.X(ref[i].x);
      var rows = vis.map(function (m) { return { m: m, q: m.points[i] }; }).sort(function (a, b) { return b.q.y - a.q.y; });
      tip.innerHTML = "<b>Bin centred at " + ref[i].x.toFixed(1) + " px</b>" +
        rows.map(function (r) { return tipRow(r.m.name, r.q.y.toFixed(3), "[" + r.q.lo.toFixed(3) + ", " + r.q.hi.toFixed(3) + "], n = " + r.q.n); }).join("");
      tip.hidden = false; place(tip, host, x / W * host.clientWidth);
      cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    }, function () { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    var leg = document.getElementById("bh-legend");
    if (leg) {
      leg.innerHTML = models.map(function (m) {
        var off = !!BH.hidden[m.name];
        return '<button type="button" class="leg leg-btn' + (off ? " is-off" : "") + '" data-name="' + esc(m.name) + '" aria-pressed="' + (!off) + '">' +
               swatch(m.name) + esc(m.name) + "</button>";
      }).join("");
      leg.querySelectorAll(".leg-btn").forEach(function (b) {
        b.addEventListener("click", function () {
          var n = b.getAttribute("data-name");
          BH.hidden[n] = !BH.hidden[n];
          if (models.every(function (m) { return BH.hidden[m.name]; })) BH.hidden[n] = false;
          renderByHeight(data);
        });
      });
    }
  }
  function initByHeight(data) {
    var seg = document.getElementById("bh-mode");
    if (seg && !seg.dataset.bound) {
      seg.dataset.bound = "1";
      seg.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () {
          BH.mode = b.getAttribute("data-mode"); BH.hidden = {};
          seg.querySelectorAll("button").forEach(function (x) {
            var on = x === b; x.classList.toggle("is-active", on); x.setAttribute("aria-pressed", on);
          });
          renderByHeight(data);
        });
      });
    }
    renderByHeight(data);
  }

  /* ======================= 2. Heatmap ======================= */
  var HM = { sort: 0, dir: -1 };
  function heatColor(v, lo, hi) {
    var t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
    /* single-hue sequential (blue), light to dark */
    var a = [239, 245, 252], b = [16, 72, 140];
    var c = a.map(function (x, i) { return Math.round(x + (b[i] - x) * t); });
    return { bg: "rgb(" + c.join(",") + ")", ink: t > 0.55 ? "#ffffff" : "#16191d" };
  }
  function renderHeatmap(data) {
    var host = document.getElementById("hm-table");
    if (!host || !data || !data.models) return;
    var cols = data.columns;
    var all = [];
    data.models.forEach(function (m) { m.values.forEach(function (v) { if (v != null) all.push(v); }); });
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var best = cols.map(function (c, j) {
      return Math.max.apply(null, data.models.map(function (m) { return m.values[j] == null ? -1 : m.values[j]; }));
    });
    var rows = data.models.slice().sort(function (a, b) {
      var x = a.values[HM.sort], y = b.values[HM.sort];
      if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
      return HM.dir < 0 ? y - x : x - y;
    });
    var h = ['<table class="heat"><thead><tr><th class="hm-model" scope="col">Model</th>'];
    cols.forEach(function (c, j) {
      var on = j === HM.sort;
      h.push('<th scope="col" aria-sort="' + (on ? (HM.dir < 0 ? "descending" : "ascending") : "none") + '"><button type="button" class="hm-sort' +
             (on ? " is-on" : "") + '" data-j="' + j + '">' + esc(c.name) + (on ? (HM.dir < 0 ? " ↓" : " ↑") : "") +
             '<span class="th-sub">n = ' + (c.n != null ? c.n.toLocaleString("en-US") : "&ndash;") + "</span></button></th>");
    });
    h.push("</tr></thead><tbody>");
    rows.forEach(function (m) {
      var ours = m.name.indexOf("Hawkeye") === 0;
      h.push('<tr class="' + (ours ? "is-ours" : "") + '"><th scope="row" class="hm-model"><span class="mname">' + esc(m.name) + "</span></th>");
      m.values.forEach(function (v, j) {
        if (v == null) { h.push('<td class="hm-na" title="not evaluated">&ndash;</td>'); return; }
        var c = heatColor(v, lo, hi), isBest = Math.abs(v - best[j]) < 1e-9;
        h.push('<td class="' + (isBest ? "hm-best" : "") + '" style="background:' + c.bg + ";color:" + c.ink + '" title="' + esc(m.name) + " · " +
               esc(cols[j].name) + ": BoW F1 " + v.toFixed(3) + (isBest ? " (best in column)" : "") + '">' + v.toFixed(2).replace(/^0/, "") + "</td>");
      });
      h.push("</tr>");
    });
    h.push("</tbody></table>");
    host.innerHTML = h.join("");
    host.querySelectorAll(".hm-sort").forEach(function (b) {
      b.addEventListener("click", function () {
        var j = +b.getAttribute("data-j");
        if (HM.sort === j) HM.dir = -HM.dir; else { HM.sort = j; HM.dir = -1; }
        renderHeatmap(data);
      });
    });
    var key = document.getElementById("hm-key");
    if (key) {
      var stops = [0, 0.25, 0.5, 0.75, 1].map(function (t) { return heatColor(lo + t * (hi - lo), lo, hi).bg; });
      key.innerHTML = '<span class="hm-key-lbl">' + lo.toFixed(2) + '</span><span class="hm-key-bar" style="background:linear-gradient(90deg,' +
                      stops.join(",") + ')"></span><span class="hm-key-lbl">' + hi.toFixed(2) + '</span><span class="hm-key-best"></span>best in column';
    }
  }

  /* ======================= 3. Forest ======================= */
  function renderForest(data) {
    var host = document.getElementById("forest-chart");
    if (!host || !data || !data.rows) return;
    var byCol = {};
    data.rows.forEach(function (r) { (byCol[r.column] = byCol[r.column] || {})[r.size] = r; });
    var cols = Object.keys(byCol);
    var mean = function (c) { var v = byCol[c]; return ((v["4B"] ? v["4B"].diff_pp : 0) + (v["8B"] ? v["8B"].diff_pp : 0)) / 2; };
    cols.sort(function (a, b) {
      var ba = a.indexOf("BSTRB") === 0, bb = b.indexOf("BSTRB") === 0;
      if (ba !== bb) return ba ? -1 : 1;
      return mean(b) - mean(a);
    });
    var W = Math.max(300, Math.min(900, host.clientWidth || 760)), narrow = W < 560;
    var labW = narrow ? 92 : 128, mr = 16, mt = 22, rowH = narrow ? 30 : 32, mb = 40;
    var H = mt + cols.length * rowH + mb;
    var lo = Math.min.apply(null, data.rows.map(function (r) { return r.lo_pp; })), hi = Math.max.apply(null, data.rows.map(function (r) { return r.hi_pp; }));
    lo = Math.floor(Math.min(lo, 0) / 5) * 5; hi = Math.ceil(hi / 5) * 5;
    var X = function (v) { return labW + (v - lo) / (hi - lo) * (W - labW - mr); };
    var s = ['<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" role="img" aria-label="Hawkeye minus base BoW F1 per dataset with 95% confidence intervals">'];
    niceTicks(lo, hi, narrow ? 4 : 7).forEach(function (v) {
      s.push('<line class="gridline" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + mt + '" y2="' + (H - mb) + '"/>');
      s.push('<text class="axis" x="' + X(v) + '" y="' + (H - mb + 15) + '" text-anchor="middle">' + (v > 0 ? "+" : "") + v + "</text>");
    });
    s.push('<line class="zero" x1="' + X(0) + '" x2="' + X(0) + '" y1="' + (mt - 6) + '" y2="' + (H - mb) + '"/>');
    s.push('<text class="ctitle" x="' + (labW + (W - labW - mr) / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Hawkeye − base, BoW F1 (percentage points)</text>');
    var nB = cols.filter(function (c) { return c.indexOf("BSTRB") === 0; }).length;
    if (nB && nB < cols.length) s.push('<line class="divider" x1="4" x2="' + (W - mr) + '" y1="' + (mt + nB * rowH) + '" y2="' + (mt + nB * rowH) + '"/>');
    cols.forEach(function (c, i) {
      var y = mt + i * rowH + rowH / 2;
      if (i % 2 === 0) s.push('<rect class="band-row" x="0" y="' + (mt + i * rowH) + '" width="' + W + '" height="' + rowH + '"/>');
      s.push('<text class="lbl" x="' + (labW - 10) + '" y="' + (y + 4) + '" text-anchor="end">' + esc(c) + "</text>");
      [["4B", -6], ["8B", 6]].forEach(function (sz) {
        var r = byCol[c][sz[0]];
        if (!r) return;
        var st = sty("Hawkeye " + sz[0]), yy = y + sz[1], sig = r.verdict !== "n.s.";
        s.push('<line x1="' + X(r.lo_pp) + '" x2="' + X(r.hi_pp) + '" y1="' + yy + '" y2="' + yy + '" style="stroke:' + st.c + ';stroke-width:2;stroke-linecap:round"/>');
        s.push('<circle cx="' + X(r.diff_pp) + '" cy="' + yy + '" r="4.6" style="fill:' + (sig ? st.c : "var(--surface)") + ";stroke:" + st.c + ';stroke-width:2"/>');
      });
      s.push('<rect class="hit" data-i="' + i + '" x="0" y="' + (mt + i * rowH) + '" width="' + W + '" height="' + rowH + '" tabindex="0" aria-label="' + esc(c) + '"/>');
    });
    s.push("</svg>");
    host.innerHTML = s.join("") + '<div class="tip" id="forest-tip" hidden></div>';
    var tip = document.getElementById("forest-tip");
    bindHits(host, function (i, rect) {
      var c = cols[+i];
      tip.innerHTML = "<b>" + esc(c) + "</b>" + ["4B", "8B"].map(function (sz) {
        var r = byCol[c][sz];
        if (!r) return "";
        return '<div class="tip-block">' + tipRow("Hawkeye " + sz, pp(r.diff_pp) + " pp", "[" + pp(r.lo_pp) + ", " + pp(r.hi_pp) + "]") +
               '<div class="tip-sub">' + f3(r.base) + " → " + f3(r.hawkeye) + " · Holm p " + (r.p_holm < 0.001 ? "&lt; 0.001" : r.p_holm.toFixed(3)) +
               " · " + (r.verdict === "n.s." ? "not significant" : r.verdict) + " · n = " + r.n.toLocaleString("en-US") + "</div></div>";
      }).join("");
      tip.hidden = false;
      tip.style.top = Math.max(8, (+rect.getAttribute("y") + rowH) / H * host.clientHeight) + "px";
      place(tip, host, X(byCol[c]["4B"] ? byCol[c]["4B"].hi_pp : 0) / W * host.clientWidth);
    }, function () { tip.hidden = true; tip.style.top = ""; });
  }

  /* ======================= 4. Smart-glasses captures ======================= */
  var OOD = { size: "8B", variant: "Degenerate removed" };
  function renderOod(data) {
    var host = document.getElementById("ood-chart");
    if (!host || !data || !data.rows) return;
    var rows = data.rows.filter(function (r) { return r.size === OOD.size && r.variant === OOD.variant; });
    var W = Math.max(300, Math.min(820, host.clientWidth || 700)), narrow = W < 560;
    var labW = narrow ? 104 : 140, deltaW = narrow ? 96 : 150, mt = 16, rowH = 62, mb = 40;
    var H = mt + rows.length * rowH + mb;
    var X = function (v) { return labW + v * (W - labW - deltaW - 10); };
    var hz = sty("Hawkeye " + OOD.size), bz = sty("Qwen3-VL-" + OOD.size);
    var s = ['<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" role="img" aria-label="Base model and Hawkeye on smart-glasses captures">'];
    [0, 0.2, 0.4, 0.6, 0.8, 1].forEach(function (v) {
      s.push('<line class="gridline" x1="' + X(v) + '" x2="' + X(v) + '" y1="' + mt + '" y2="' + (H - mb) + '"/>');
      s.push('<text class="axis" x="' + X(v) + '" y="' + (H - mb + 15) + '" text-anchor="middle">' + v.toFixed(1) + "</text>");
    });
    s.push('<text class="ctitle" x="' + (labW + (W - labW - deltaW) / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Score (mean over images)</text>');
    s.push('<text class="ctitle" x="' + (W - 4) + '" y="' + (mt + 2) + '" text-anchor="end">Δ pp [95% CI]</text>');
    rows.forEach(function (r, i) {
      var y = mt + i * rowH + rowH / 2 + 6;
      s.push('<text class="lbl" x="' + (labW - 12) + '" y="' + (y + 4) + '" text-anchor="end">' + esc(r.metric) + "</text>");
      s.push('<line x1="' + X(r.base) + '" x2="' + X(r.hawkeye) + '" y1="' + y + '" y2="' + y + '" style="stroke:var(--ink-3);stroke-width:2;opacity:.5"/>');
      s.push('<circle cx="' + X(r.base) + '" cy="' + y + '" r="7" style="fill:var(--surface);stroke:' + bz.c + ';stroke-width:2.4"/>');
      s.push('<circle cx="' + X(r.hawkeye) + '" cy="' + y + '" r="7" style="fill:' + hz.c + ';stroke:var(--surface);stroke-width:2"/>');
      s.push('<text class="val" x="' + X(r.base) + '" y="' + (y + 22) + '" text-anchor="middle">' + r.base.toFixed(3) + "</text>");
      s.push('<text class="val strong" x="' + X(r.hawkeye) + '" y="' + (y - 13) + '" text-anchor="middle">' + r.hawkeye.toFixed(3) + "</text>");
      s.push('<text class="val ' + (r.significant ? "sig" : "ns") + '" x="' + (W - 4) + '" y="' + (y + 4) + '" text-anchor="end">' +
             (r.diff_pp >= 0 ? "+" : "−") + Math.abs(r.diff_pp).toFixed(1) + (narrow ? "" : " [" + r.lo_pp.toFixed(1) + ", " + r.hi_pp.toFixed(1) + "]") +
             (r.significant ? "" : " n.s.") + "</text>");
      s.push('<rect class="hit" data-i="' + i + '" x="0" y="' + (mt + i * rowH) + '" width="' + W + '" height="' + rowH + '" tabindex="0" aria-label="' + esc(r.metric) + '"/>');
    });
    s.push("</svg>");
    host.innerHTML = s.join("") + '<div class="tip" id="ood-tip" hidden></div>';
    var tip = document.getElementById("ood-tip");
    bindHits(host, function (i, rect) {
      var r = rows[+i];
      tip.innerHTML = "<b>" + esc(r.metric) + " · " + esc(r.variant) + "</b>" +
        tipRow("Qwen3-VL-" + r.size + " (base)", r.base.toFixed(3)) + tipRow("Hawkeye " + r.size, r.hawkeye.toFixed(3)) +
        '<div class="tip-sub">Δ ' + pp(r.diff_pp) + " pp [" + pp(r.lo_pp) + ", " + pp(r.hi_pp) + "] · Holm p " +
        (r.p_holm < 0.001 ? "&lt; 0.001" : r.p_holm.toFixed(3)) + (r.significant ? "" : " · not significant") + "</div>";
      tip.hidden = false;
      tip.style.top = ((+rect.getAttribute("y") + rowH) / H * host.clientHeight) + "px";
      place(tip, host, X(Math.max(r.base, r.hawkeye)) / W * host.clientWidth);
    }, function () { tip.hidden = true; tip.style.top = ""; });
    var leg = document.getElementById("ood-legend");
    if (leg) leg.innerHTML = '<span class="leg"><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6" style="fill:var(--surface);stroke:' +
      bz.c + ';stroke-width:2.2"/></svg>Qwen3-VL-' + OOD.size + ' (base)</span><span class="leg"><svg width="16" height="16" aria-hidden="true"><circle cx="8" cy="8" r="6.5" style="fill:' +
      hz.c + '"/></svg>Hawkeye ' + OOD.size + "</span>";
  }
  function initOod(data) {
    [["ood-size", "size"], ["ood-variant", "variant"]].forEach(function (p) {
      var seg = document.getElementById(p[0]);
      if (!seg || seg.dataset.bound) return;
      seg.dataset.bound = "1";
      seg.querySelectorAll("button").forEach(function (b) {
        b.addEventListener("click", function () {
          OOD[p[1]] = b.getAttribute("data-v");
          seg.querySelectorAll("button").forEach(function (x) { var on = x === b; x.classList.toggle("is-active", on); x.setAttribute("aria-pressed", on); });
          renderOod(data);
        });
      });
    });
    renderOod(data);
  }

  /* ======================= 5. OCR engines vs. VLMs ======================= */
  function renderOcr(data) {
    var host = document.getElementById("ocr-chart");
    if (!host || !data || !data.curves) return;
    var curves = data.curves;
    var W = Math.max(300, Math.min(860, host.clientWidth || 700)), narrow = W < 560;
    var H = narrow ? 280 : 310, ml = 44, mr = 14, mt = 14, mb = 44;
    var xs = curves[0].x, xmax = Math.ceil(Math.max.apply(null, xs) / 5) * 5;
    var X = function (x) { return ml + x / xmax * (W - ml - mr); };
    var Y = function (v) { return mt + (1 - v) * (H - mt - mb); };
    var s = ['<svg viewBox="0 0 ' + W + " " + H + '" width="' + W + '" role="img" aria-label="BoW F1 by text height for OCR engines and VLMs">'];
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      s.push('<line class="gridline" x1="' + ml + '" x2="' + (W - mr) + '" y1="' + Y(v) + '" y2="' + Y(v) + '"/>');
      s.push('<text class="axis" x="' + (ml - 7) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + v.toFixed(2) + "</text>");
    });
    niceTicks(0, xmax, narrow ? 5 : 9).forEach(function (x) {
      s.push('<text class="axis" x="' + X(x) + '" y="' + (H - mb + 16) + '" text-anchor="middle">' + x + "</text>");
    });
    s.push('<text class="ctitle" x="' + (ml + (W - ml - mr) / 2) + '" y="' + (H - 6) + '" text-anchor="middle">Mean text height (px)</text>');
    curves.forEach(function (c) {
      var st = sty(c.name);
      if (c.lo && c.lo.length === c.x.length) {
        var up = c.x.map(function (x, i) { return X(x) + "," + Y(c.hi[i]); });
        var dn = c.x.map(function (x, i) { return X(x) + "," + Y(c.lo[i]); }).reverse();
        s.push('<polygon points="' + up.concat(dn).join(" ") + '" style="fill:' + st.c + ";opacity:" + (st.band ? 0.14 : 0.07) + '"/>');
      }
    });
    curves.forEach(function (c) {
      var st = sty(c.name);
      s.push('<polyline points="' + c.x.map(function (x, i) { return X(x).toFixed(1) + "," + Y(c.y[i]).toFixed(1); }).join(" ") +
             '" fill="none" stroke-linejoin="round" stroke-linecap="round" style="stroke:' + st.c + ";stroke-width:" + st.w +
             (st.d ? ";stroke-dasharray:" + st.d : "") + '"/>');
      if (st.dot) c.x.forEach(function (x, i) { s.push('<circle cx="' + X(x) + '" cy="' + Y(c.y[i]) + '" r="3.6" style="fill:' + st.c + ';stroke:var(--surface);stroke-width:1.8"/>'); });
    });
    s.push('<line id="ocr-cross" class="crosshair" x1="0" x2="0" y1="' + mt + '" y2="' + (H - mb) + '" visibility="hidden"/>');
    xs.forEach(function (x, i) {
      var xl = i ? (X(xs[i - 1]) + X(x)) / 2 : ml, xr = i < xs.length - 1 ? (X(xs[i + 1]) + X(x)) / 2 : W - mr;
      s.push('<rect class="hit" data-i="' + i + '" x="' + xl + '" y="' + mt + '" width="' + (xr - xl) + '" height="' + (H - mt - mb) + '" tabindex="0" aria-label="' + x.toFixed(1) + ' px"/>');
    });
    s.push("</svg>");
    host.innerHTML = s.join("") + '<div class="tip" id="ocr-tip" hidden></div>';
    var tip = document.getElementById("ocr-tip"), cross = document.getElementById("ocr-cross");
    bindHits(host, function (i) {
      i = +i;
      var rows = curves.map(function (c) { return { c: c, v: c.y[i] }; }).sort(function (a, b) { return b.v - a.v; });
      tip.innerHTML = "<b>Bin centred at " + xs[i].toFixed(1) + " px</b>" + rows.map(function (r) {
        var ci = r.c.lo && r.c.lo.length ? "[" + r.c.lo[i].toFixed(3) + ", " + r.c.hi[i].toFixed(3) + "]" : "";
        return tipRow(r.c.name, r.v.toFixed(3), ci);
      }).join("");
      tip.hidden = false; place(tip, host, X(xs[i]) / W * host.clientWidth);
      cross.setAttribute("x1", X(xs[i])); cross.setAttribute("x2", X(xs[i])); cross.setAttribute("visibility", "visible");
    }, function () { tip.hidden = true; cross.setAttribute("visibility", "hidden"); });
    var leg = document.getElementById("ocr-legend");
    if (leg) leg.innerHTML = curves.map(function (c) { return '<span class="leg">' + swatch(c.name) + esc(c.name) + "</span>"; }).join("");

    /* sharp reference frames: bars with 95% CI */
    var bh = document.getElementById("ocr-sharp");
    if (!bh || !data.sharp_frames) return;
    var items = Object.keys(data.sharp_frames).map(function (k) { var v = data.sharp_frames[k]; return { name: k, v: v.bow_f1, lo: v.lo, hi: v.hi, n: v.n }; })
      .sort(function (a, b) { return b.v - a.v; });
    var BW = Math.max(300, Math.min(860, bh.clientWidth || 700)), bn = BW < 560;
    var lab = bn ? 128 : 170, valW = 48, top = 8, rh = 28, bot = 34, BH2 = top + items.length * rh + bot;
    var BX = function (v) { return lab + v * (BW - lab - valW); };
    var b = ['<svg viewBox="0 0 ' + BW + " " + BH2 + '" width="' + BW + '" role="img" aria-label="BoW F1 on sharp reference frames">'];
    [0, 0.25, 0.5, 0.75, 1].forEach(function (v) {
      b.push('<line class="gridline" x1="' + BX(v) + '" x2="' + BX(v) + '" y1="' + top + '" y2="' + (BH2 - bot) + '"/>');
      b.push('<text class="axis" x="' + BX(v) + '" y="' + (BH2 - bot + 15) + '" text-anchor="middle">' + v.toFixed(2) + "</text>");
    });
    b.push('<text class="ctitle" x="' + (lab + (BW - lab - valW) / 2) + '" y="' + (BH2 - 4) + '" text-anchor="middle">BoW F1 on the ' + items[0].n + " sharp reference frames</text>");
    items.forEach(function (it, i) {
      var y = top + i * rh, st = sty(it.name), ours = it.name.indexOf("Hawkeye") === 0;
      b.push('<text class="lbl' + (ours ? " strong" : "") + '" x="' + (lab - 10) + '" y="' + (y + rh / 2 + 4) + '" text-anchor="end">' + esc(it.name) + "</text>");
      b.push('<rect x="' + lab + '" y="' + (y + 7) + '" width="' + (BX(it.v) - lab) + '" height="' + (rh - 14) + '" rx="3" style="fill:' + st.c + ";opacity:" + (st.kind === "engine" ? 0.75 : 0.9) + '"/>');
      b.push('<line x1="' + BX(it.lo) + '" x2="' + BX(it.hi) + '" y1="' + (y + rh / 2) + '" y2="' + (y + rh / 2) + '" style="stroke:var(--ink);stroke-width:1.4"/>');
      b.push('<text class="val" x="' + (BX(it.hi) + 6) + '" y="' + (y + rh / 2 + 4) + '">' + it.v.toFixed(3) + "</text>");
      b.push('<rect class="hit" data-i="' + i + '" x="0" y="' + y + '" width="' + BW + '" height="' + rh + '" tabindex="0" aria-label="' + esc(it.name) + '"/>');
    });
    b.push("</svg>");
    bh.innerHTML = b.join("") + '<div class="tip" id="ocr-sharp-tip" hidden></div>';
    var t2 = document.getElementById("ocr-sharp-tip");
    bindHits(bh, function (i, rect) {
      var it = items[+i];
      t2.innerHTML = "<b>" + esc(it.name) + "</b>" + tipRow(it.name, it.v.toFixed(3), "[" + it.lo.toFixed(3) + ", " + it.hi.toFixed(3) + "], n = " + it.n);
      t2.hidden = false; t2.style.top = ((+rect.getAttribute("y") + rh) / BH2 * bh.clientHeight) + "px";
      place(t2, bh, BX(it.hi) / BW * bh.clientWidth);
    }, function () { t2.hidden = true; t2.style.top = ""; });
  }

  /* ======================= 6. Snellen example chart ======================= */
  function charDiff(gt, pred) {
    var out = [], n = Math.max(gt.length, pred.length);
    for (var i = 0; i < n; i++) {
      var g = gt[i], p = pred[i];
      if (p === undefined) out.push('<span class="ch bad miss" title="missing">·</span>');
      else out.push('<span class="ch' + (p === g ? "" : " bad") + '">' + esc(p) + "</span>");
    }
    return out.join("");
  }
  function renderSnellenExample(data) {
    var rows = document.getElementById("snx-rows");
    if (rows && data.rows && data.heights_px) {
      rows.innerHTML = data.rows.map(function (r, i) {
        return '<tr' + (i >= data.rows.length - 2 ? ' class="hl"' : "") + "><td>" + (i + 1) + "</td><td>" + data.heights_px[i] +
               ' px</td><td class="mono">' + esc(r) + "</td></tr>";
      }).join("");
    }
    var tb = document.querySelector("#snx-readings tbody");
    if (!tb || !data.models) return;
    var ref = data.models[0].small_rows.map(function (r) { return r.gt; });
    var head = document.querySelector("#snx-readings thead tr");
    if (head) head.innerHTML = '<th>Model</th>' + ref.map(function (g, k) {
      return '<th class="txt">Row ' + (data.rows.length - ref.length + k + 1) + " (" + data.heights_px[data.rows.length - ref.length + k] + ' px)<span class="th-sub mono">' + esc(g) + "</span></th><th>Acc.</th>";
    }).join("");
    tb.innerHTML = data.models.map(function (m) {
      var ours = m.name.indexOf("Hawkeye") === 0;
      return '<tr class="' + (ours ? "is-ours" : "") + '"><td><span class="mname">' + esc(m.name) + "</span></td>" + m.small_rows.map(function (r) {
        return '<td class="txt mono">' + charDiff(r.gt, r.pred) + "</td><td>" + r.accuracy.toFixed(2) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    var note = document.getElementById("snx-note");
    if (note && data.note) note.textContent = data.note;
  }

  function boot() {
    load("by-height-data", "static/data/by_height.json", initByHeight);
    load("heatmap-data", "static/data/heatmap.json", function (d) { if (!document.querySelector("#hm-table table")) renderHeatmap(d); });
    load("forest-data", "static/data/forest.json", renderForest);
    load("ood-data", "static/data/ood.json", initOod);
    load("ocr-data", "static/data/ocr.json", renderOcr);
    load("snellen-example-data", "static/data/snellen_example.json", function (d) { if (!document.querySelector("#snx-readings tbody tr td.mono")) renderSnellenExample(d); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();

/* ======================= 7. Attention viewer ======================= */
/* Per-arm attention maps (uint8 grids, base64) drawn over the frame on a canvas, either the mean
   over decoding steps or one step at a time with a shared slider. Loaded with fetch, so it needs
   the page to be served over HTTP (as on GitHub Pages). */
(function () {
  "use strict";
  var root = document.getElementById("attn");
  if (!root) return;
  var grid = document.getElementById("attn-grid"), frBar = document.getElementById("attn-frames");
  var modeBar = document.getElementById("attn-mode"), ctr = document.getElementById("attn-controls");
  var slider = document.getElementById("attn-step"), count = document.getElementById("attn-count"), play = document.getElementById("attn-play");
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var S = { frame: null, data: null, mode: "mean", step: 0, n: 0, timer: null, cache: {}, img: null };

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function decode(b64) { var s = atob(b64), a = new Uint8Array(s.length); for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
  /* low = transparent, high = orange to yellow */
  var LUT = (function () {
    var stops = [[0, 255, 115, 0, 0], [0.35, 255, 115, 0, 110], [0.7, 255, 200, 30, 210], [1, 255, 255, 150, 240]], out = [];
    for (var v = 0; v < 256; v++) {
      var t = v / 255, k = 0;
      while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
      var a = stops[k], b = stops[k + 1], u = (t - a[0]) / (b[0] - a[0]);
      out.push([0, 1, 2, 3].map(function (j) { return Math.round(a[j + 1] + (b[j + 1] - a[j + 1]) * u); }));
    }
    return out;
  })();
  function heatCanvas(bytes, gy, gx) {
    var c = document.createElement("canvas"); c.width = gx; c.height = gy;
    var ctx = c.getContext("2d"), im = ctx.createImageData(gx, gy);
    for (var i = 0; i < gy * gx; i++) { var col = LUT[bytes[i]]; im.data.set(col, i * 4); }
    ctx.putImageData(im, 0, 0);
    return c;
  }
  function stop() { if (S.timer) { clearInterval(S.timer); S.timer = null; } play.innerHTML = "&#9654;"; play.setAttribute("aria-label", "Play"); }
  function tokText(tokens, upto, hl) {
    var parts = [];
    for (var i = 0; i <= upto && i < tokens.length; i++) {
      var t = tokens[i].replace(/\n/g, " · ");
      parts.push(i === hl ? "<mark>" + esc(t) + "</mark>" : esc(t));
    }
    var txt = parts.join(""), plain = txt.replace(/<[^>]+>/g, "");
    if (plain.length > 150) {          /* keep the end: the current token and what precedes it */
      var cut = 0, seen = 0;
      while (cut < txt.length && plain.length - seen > 150) { if (txt[cut] === "<") { cut = txt.indexOf(">", cut) + 1; continue; } seen++; cut++; }
      txt = "…" + txt.slice(cut);
    }
    return txt;
  }
  function draw() {
    var d = S.data;
    if (!d) return;
    d.arms.forEach(function (arm, k) {
      var panel = grid.children[k];
      if (!panel) return;
      var cv = panel.querySelector("canvas"), stage = panel.querySelector(".attn-stage");
      var w = stage.clientWidth, h = Math.round(w * d.h / d.w);
      var dpr = window.devicePixelRatio || 1;
      cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); cv.style.height = h + "px";
      var ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
      ctx.drawImage(S.img, 0, 0, w, h);
      ctx.fillStyle = "rgba(0,0,0,.28)"; ctx.fillRect(0, 0, w, h);          /* dim the photo slightly */
      var n = arm.maps.length, bytes, done = false, i = S.step;
      if (S.mode === "mean") bytes = arm._mean || (arm._mean = decode(arm.mean));
      else { done = i >= n; var j = Math.min(i, n - 1); arm._maps = arm._maps || []; bytes = arm._maps[j] || (arm._maps[j] = decode(arm.maps[j])); }
      ctx.drawImage(heatCanvas(bytes, arm.grid[0], arm.grid[1]), 0, 0, w, h);
      panel.classList.toggle("is-done", done);
      var txt = panel.querySelector(".attn-text"), meta = panel.querySelector(".attn-meta");
      if (S.mode === "mean") {
        txt.innerHTML = "<span>" + tokText(arm.tokens, arm.tokens.length - 1, -1) + "</span>";
        meta.textContent = arm.tokens.length + " tokens" + (arm.truncated ? " (first 160 tokens)" : "");
      } else {
        var hi = Math.min(i, n - 1);
        txt.innerHTML = "<span>" + tokText(arm.tokens, hi, done ? -1 : hi) + "</span>";
        meta.textContent = done ? "finished after " + n + " tokens" : "step " + (i + 1) + " / " + n + (arm.truncated ? " (first 160 tokens)" : "");
      }
    });
    if (S.mode === "step") count.textContent = "Step " + (S.step + 1) + " / " + S.n;
  }
  function build() {
    var d = S.data;
    grid.innerHTML = d.arms.map(function (arm) {
      var ours = arm.name.indexOf("Hawkeye") === 0;
      return '<div class="attn-panel' + (ours ? " is-ours" : "") + '"><div class="attn-title"><span class="mname">' + esc(arm.name) + "</span></div>" +
             '<div class="attn-stage"><canvas role="img" aria-label="Attention of ' + esc(arm.name) + ' over the frame"></canvas></div>' +
             '<div class="attn-meta"></div><div class="attn-text mono"></div></div>';
    }).join("");
    S.n = Math.max.apply(null, d.arms.map(function (a) { return a.maps.length; }));
    slider.max = S.n - 1; S.step = Math.min(S.step, S.n - 1); slider.value = S.step;
    draw();
  }
  function loadFrame(id) {
    stop();
    S.frame = id;
    frBar.querySelectorAll("button").forEach(function (b) { var on = b.getAttribute("data-v") === id; b.classList.toggle("is-active", on); b.setAttribute("aria-pressed", on); });
    var go = function (d) {
      var img = new Image();
      img.onload = function () { S.data = d; S.img = img; S.step = 0; build(); };
      img.src = d.image;
    };
    if (S.cache[id]) { go(S.cache[id]); return; }
    fetch("static/attention/" + id + ".json", { cache: "no-cache" }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (d) { S.cache[id] = d; go(d); })
      .catch(function () { grid.innerHTML = '<p class="muted">The attention data could not be loaded.</p>'; });
  }
  function setMode(m) {
    S.mode = m; stop();
    modeBar.querySelectorAll("button").forEach(function (b) { var on = b.getAttribute("data-v") === m; b.classList.toggle("is-active", on); b.setAttribute("aria-pressed", on); });
    ctr.hidden = m !== "step";
    draw();
  }
  modeBar.querySelectorAll("button").forEach(function (b) { b.addEventListener("click", function () { setMode(b.getAttribute("data-v")); }); });
  slider.addEventListener("input", function () { stop(); S.step = +slider.value; draw(); });
  play.addEventListener("click", function () {
    if (S.timer) { stop(); return; }
    play.innerHTML = "&#10074;&#10074;"; play.setAttribute("aria-label", "Pause");
    S.timer = setInterval(function () { S.step = (S.step + 1) % S.n; slider.value = S.step; draw(); }, 166);
  });
  var rt = null;
  window.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(draw, 150); });

  if (location.protocol === "file:" || !window.fetch) {
    grid.innerHTML = '<p class="muted">Open the page over HTTP to load the attention viewer.</p>';
    return;
  }
  fetch("static/attention/index.json", { cache: "no-cache" }).then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (ix) {
      frBar.innerHTML = ix.frames.map(function (f) { return '<button type="button" class="seg-btn" data-v="' + esc(f.id) + '" aria-pressed="false">' + esc(f.label) + "</button>"; }).join("");
      frBar.querySelectorAll("button").forEach(function (b) { b.addEventListener("click", function () { loadFrame(b.getAttribute("data-v")); }); });
      var note = document.getElementById("attn-note");
      if (note && ix.note) note.textContent = ix.note + " Use “Per decoding step” to follow the attention as each token is generated.";
      if (reduce) play.title = "Animation is paused by default (reduced motion).";
      loadFrame(ix.frames[0].id);
    })
    .catch(function () { grid.innerHTML = '<p class="muted">Open the page over HTTP to load the attention viewer.</p>'; });
})();
