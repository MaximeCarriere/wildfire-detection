/* Pruning course — interactive pieces.
   No dependencies, no build step, no network calls. Safe under a strict CSP
   as long as it is loaded as an external file (which is why it lives here
   rather than inline).

   Three widgets:
     1. granularity explorer  — the same 50% removed in five different patterns
     2. criterion explorer    — which channels each importance rule would delete
     3. two charts            — committed measurements, hover + table view

   Colours come from CSS custom properties, so both themes are handled by the
   stylesheet and nothing here needs to know which one is active. */
(function () {
  "use strict";

  var tip = document.getElementById("tip");
  function showTip(html, x, y) {
    if (!tip) return;
    tip.innerHTML = html;
    tip.classList.add("on");
    var r = tip.getBoundingClientRect();
    var left = x + 14, top = y - r.height - 12;
    if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
    if (top < 8) top = y + 18;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function hideTip() { if (tip) tip.classList.remove("on"); }

  /* ---------------- 1. granularity ----------------
     A 12x12 weight grid read as 4 output channels x 4 input channels, each
     cell of that being one 3x3 convolution kernel. Every pattern below removes
     exactly half the weights; only the SHAPE of the removal changes, which is
     the entire point of the granularity axis. */
  var N = 12, KS = 3, NK = 4;
  var GRAN = {
    fine: {
      label: "fine-grained",
      what: "<strong>Individual weights</strong>, anywhere, with no pattern at all.",
      why: "Maximum freedom gives the highest compression ratio of any method — historically 9&times; to 12&times; on classic networks — because you can always find the genuinely useless weights.",
      cls: "no", hwLabel: "no speed-up",
      hw: "Nothing standard. Scattered zeros still sit inside a full-size matrix, so a GPU does exactly the same work.",
      mask: function () {
        // Fixed seed: the scatter should look arbitrary but stay put between clicks.
        var out = {}, s = 7, cells = [], i, j, t;
        for (i = 0; i < N * N; i++) cells.push(i);
        for (i = cells.length - 1; i > 0; i--) {
          s = (s * 1103515245 + 12345) & 0x7fffffff;
          j = s % (i + 1); t = cells[i]; cells[i] = cells[j]; cells[j] = t;
        }
        for (i = 0; i < (N * N) / 2; i++) out[cells[i]] = 1;
        return out;
      }
    },
    nm: {
      label: "2:4 pattern",
      what: "<strong>Exactly two of every four neighbouring weights</strong> in a row. The count is fixed; only the choice inside each group is free.",
      why: "Rigid enough for silicon to exploit, loose enough to still drop the smaller weight in each group. Fixed 50% sparsity.",
      cls: "yes", hwLabel: "real speed-up",
      hw: "Ampere sparse tensor cores, up to ~2&times; peak. <strong>The target board is Ampere-class.</strong>",
      mask: function () {
        var out = {}, r, g, k, drop;
        for (r = 0; r < N; r++) for (g = 0; g < N; g += 4) {
          drop = [(r + g) % 4, (r + g + 2) % 4];
          for (k = 0; k < 2; k++) out[r * N + g + drop[k]] = 1;
        }
        return out;
      }
    },
    vector: {
      label: "vector",
      what: "<strong>Whole rows inside a kernel</strong> — a 1&times;3 strip of a 3&times;3 filter.",
      why: "A middle ground: more regular than scattered weights, finer than deleting whole filters.",
      cls: "partial", hwLabel: "partial",
      hw: "Partially. Needs kernels written for the pattern; not a general win.",
      mask: function () {
        var out = {}, kr, kc, rr, cc;
        for (kr = 0; kr < NK; kr++) for (kc = 0; kc < NK; kc++)
          for (rr = 0; rr < KS; rr++) {
            if ((kr + kc + rr) % 2 === 0) continue;
            for (cc = 0; cc < KS; cc++) out[(kr * KS + rr) * N + kc * KS + cc] = 1;
          }
        return out;
      }
    },
    kernel: {
      label: "kernel",
      what: "<strong>Entire 3&times;3 kernels</strong> — one filter's whole response to one input channel.",
      why: "Coarser again, and it removes a meaningful unit: a complete connection between two channels.",
      cls: "partial", hwLabel: "partial",
      hw: "Partially. The matrix keeps its shape, but the blocks of zeros are at least regular.",
      mask: function () {
        var out = {}, kr, kc, rr, cc;
        for (kr = 0; kr < NK; kr++) for (kc = 0; kc < NK; kc++) {
          if ((kr + kc) % 2 !== 0) continue;
          for (rr = 0; rr < KS; rr++) for (cc = 0; cc < KS; cc++)
            out[(kr * KS + rr) * N + kc * KS + cc] = 1;
        }
        return out;
      }
    },
    channel: {
      label: "channel",
      what: "<strong>Whole channels</strong> — every weight feeding into or out of one channel disappears together.",
      why: "The network genuinely shrinks: fewer rows and columns, less arithmetic, a smaller file, and no special support needed anywhere.",
      cls: "yes", hwLabel: "real speed-up",
      hw: "<strong>Any hardware.</strong> The result is simply a smaller dense matrix. This is what has been tested so far.",
      mask: function () {
        var out = {}, kc, r, cc;
        for (kc = 0; kc < NK; kc++) {
          if (kc % 2 !== 0) continue;
          for (r = 0; r < N; r++) for (cc = 0; cc < KS; cc++) out[r * N + kc * KS + cc] = 1;
        }
        return out;
      }
    }
  };

  var wmat = document.getElementById("wmat");
  var granCtl = document.getElementById("gran-ctl");
  if (wmat && granCtl) {
    var cells = [], i;
    for (i = 0; i < N * N; i++) {
      var d = document.createElement("div");
      d.className = "wcell";
      wmat.appendChild(d);
      cells.push(d);
    }
    var drawGran = function (key) {
      var g = GRAN[key], cut = g.mask();
      cells.forEach(function (c, idx) {
        if (cut[idx]) c.classList.add("cut"); else c.classList.remove("cut");
      });
      document.getElementById("gran-what").innerHTML = g.what;
      document.getElementById("gran-why").innerHTML = g.why;
      document.getElementById("gran-hw").innerHTML =
        '<span class="hw ' + g.cls + '">' + g.hwLabel + "</span><div>" + g.hw + "</div>";
      Array.prototype.forEach.call(granCtl.querySelectorAll("button"), function (b) {
        b.setAttribute("aria-pressed", String(b.getAttribute("data-k") === key));
      });
    };
    Object.keys(GRAN).forEach(function (k, idx) {
      var b = document.createElement("button");
      b.textContent = GRAN[k].label;
      b.setAttribute("data-k", k);
      b.setAttribute("aria-pressed", String(idx === 0));
      b.addEventListener("click", function () { drawGran(k); });
      granCtl.appendChild(b);
    });
    drawGran("fine");
  }

  /* ---------------- 2. criterion ----------------
     Ten invented channels with three weights each, plus the extra signals the
     data-driven criteria need. The numbers are chosen so the criteria visibly
     DISAGREE: ch 3 and ch 7 each carry one large weight among near-zeros, which
     L2 rewards and L1 does not. */
  var CH = [
    { n: "ch 0", w: [0.82, 0.71, 0.66], bn: 0.91, grad: 0.12, dist: 0.74 },
    { n: "ch 1", w: [0.11, 0.09, 0.08], bn: 0.14, grad: 0.62, dist: 0.20 },
    { n: "ch 2", w: [0.55, 0.51, 0.49], bn: 0.60, grad: 0.31, dist: 0.52 },
    { n: "ch 3", w: [0.05, 0.04, 0.92], bn: 0.34, grad: 0.08, dist: 0.31 },
    { n: "ch 4", w: [0.44, 0.41, 0.40], bn: 0.47, grad: 0.77, dist: 0.42 },
    { n: "ch 5", w: [0.19, 0.17, 0.15], bn: 0.09, grad: 0.21, dist: 0.17 },
    { n: "ch 6", w: [0.68, 0.65, 0.61], bn: 0.72, grad: 0.44, dist: 0.65 },
    { n: "ch 7", w: [0.28, 0.90, 0.06], bn: 0.51, grad: 0.15, dist: 0.38 },
    { n: "ch 8", w: [0.36, 0.33, 0.31], bn: 0.39, grad: 0.55, dist: 0.34 },
    { n: "ch 9", w: [0.14, 0.12, 0.60], bn: 0.22, grad: 0.09, dist: 0.26 }
  ];
  function sumAbs(w) { return w[0] + w[1] + w[2]; }
  var CRIT = {
    l2: {
      label: "L2 magnitude",
      score: function (c) { return Math.sqrt(c.w[0] * c.w[0] + c.w[1] * c.w[1] + c.w[2] * c.w[2]); },
      note: "Ranks by the square root of summed squares. Because squaring exaggerates large values, <strong>one big weight can carry a channel</strong> whose others are near zero — look at ch 3 and ch 7. This is the criterion used in the study so far."
    },
    l1: {
      label: "L1 magnitude",
      score: function (c) { return sumAbs(c.w); },
      note: "Adds absolute values instead of squares. Treats three medium weights as more valuable than one spike plus two near-zeros — the opposite bias to L2. <strong>One character apart in code, genuinely different choices.</strong>"
    },
    bn: {
      label: "BN scale",
      score: function (c) { return c.bn; },
      note: "Reuses a number the network already computes. Every channel passes through batch normalisation, which multiplies it by a learned scale — so <strong>the network has already been telling you which channels it turns down</strong>. Free to compute, and this detector is dense with batch norm."
    },
    taylor: {
      label: "Taylor",
      score: function (c) { return c.grad * sumAbs(c.w); },
      note: "Estimates how far the <em>error</em> would move if this channel vanished, using gradients. The only family that looks at <strong>data rather than weights</strong>, so it needs a real backward pass first — and it can rank a small weight as critical when the loss is very sensitive to it."
    },
    fpgm: {
      label: "FPGM",
      score: function (c) { return Math.abs(c.dist - 0.4); },
      note: "Changes the question from \"which is smallest\" to <strong>\"which is most redundant\"</strong>. A channel sitting close to the average of all the others is duplicating work already being done, so it can go even when its weights are large."
    },
    random: {
      label: "random",
      score: function () { return Math.random(); },
      note: "<strong>The control, and the most important run in the set.</strong> If a clever criterion cannot beat this after retraining, the clever part was never doing the work. Click it again — the selection changes every time, which is exactly the point."
    }
  };
  var rowsEl = document.getElementById("chan-rows");
  var critCtl = document.getElementById("crit-ctl");
  if (rowsEl && critCtl) {
    CH.forEach(function (c) {
      var el = document.createElement("div");
      el.className = "chan";
      el.innerHTML = '<span class="chan-name">' + c.n + '</span>' +
        '<span class="chan-track"><span class="chan-fill"></span></span>' +
        '<span class="chan-val"></span>';
      rowsEl.appendChild(el);
      c.el = el;
    });
    var drawCrit = function (key) {
      var spec = CRIT[key];
      var scored = CH.map(function (c) { return { c: c, s: spec.score(c) }; });
      var max = 0;
      scored.forEach(function (x) { if (x.s > max) max = x.s; });
      if (!max) max = 1;
      var doomed = {};
      scored.slice().sort(function (a, b) { return a.s - b.s; }).slice(0, 3)
        .forEach(function (x) { doomed[x.c.n] = 1; });
      scored.forEach(function (x) {
        var cut = !!doomed[x.c.n];
        if (cut) x.c.el.classList.add("cut"); else x.c.el.classList.remove("cut");
        x.c.el.querySelector(".chan-fill").style.width = Math.max(3, (x.s / max) * 100) + "%";
        x.c.el.querySelector(".chan-val").textContent = cut ? "cut" : x.s.toFixed(2);
      });
      document.getElementById("crit-explain").innerHTML = spec.note;
      Array.prototype.forEach.call(critCtl.querySelectorAll("button"), function (b) {
        b.setAttribute("aria-pressed", String(b.getAttribute("data-k") === key));
      });
    };
    Object.keys(CRIT).forEach(function (k, idx) {
      var b = document.createElement("button");
      b.textContent = CRIT[k].label;
      b.setAttribute("data-k", k);
      b.setAttribute("aria-pressed", String(idx === 0));
      b.addEventListener("click", function () { drawCrit(k); });
      critCtl.appendChild(b);
    });
    drawCrit("l2");
  }

  /* ---------------- 3. charts ---------------- */
  function svgEl(t, a) {
    var e = document.createElementNS("http://www.w3.org/2000/svg", t), k;
    for (k in a) e.setAttribute(k, a[k]);
    return e;
  }

  /* Committed measurements: pruning damage with no retraining, full test set. */
  var DMG = [
    { x: "none", p: "7.03 M", macs: "—", all: 0.7775, small: 0.6062, tiny: 0.1386 },
    { x: "2%",   p: "6.84 M", macs: "4.1%",  all: 0.6872, small: 0.4441, tiny: 0.0650 },
    { x: "5%",   p: "6.53 M", macs: "10.4%", all: 0.0956, small: 0.0327, tiny: 0.0036 },
    { x: "10%",  p: "5.97 M", macs: "19.8%", all: 0.0052, small: 0.0000, tiny: 0.0000 },
    { x: "25%",  p: "4.24 M", macs: "43.1%", all: 0.0000, small: 0.0000, tiny: 0.0000 },
    { x: "50%",  p: "1.86 M", macs: "73.0%", all: 0.0000, small: 0.0000, tiny: 0.0000 },
    { x: "70%",  p: "0.62 M", macs: "88.9%", all: 0.0000, small: 0.0000, tiny: 0.0000 }
  ];
  /* Size tiers are nested subsets (overall contains small contains tiny), so
     this is ordinal data and takes one hue light-to-dark, not three hues. */
  var SER = [
    { k: "all",   name: "overall", v: "--seq-1" },
    { k: "small", name: "small",   v: "--seq-2" },
    { k: "tiny",  name: "tiny",    v: "--seq-3" }
  ];
  function drawDamage() {
    var host = document.getElementById("dmg-chart");
    if (!host) return;
    var W = 780, H = 340, ml = 56, mr = 96, mt = 16, mb = 46;
    var iw = W - ml - mr, ih = H - mt - mb;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Accuracy collapses to zero by a 10 percent channel cut" });
    var X = function (i) { return ml + (iw / (DMG.length - 1)) * i; };
    var Y = function (v) { return mt + ih - (v / 0.8) * ih; };

    [0, 0.2, 0.4, 0.6, 0.8].forEach(function (g) {
      svg.appendChild(svgEl("line", { x1: ml, x2: ml + iw, y1: Y(g), y2: Y(g),
        stroke: "var(--hair)", "stroke-width": 1 }));
      var t = svgEl("text", { x: ml - 11, y: Y(g) + 4, "text-anchor": "end",
        fill: "var(--frontBrightText)", "font-size": 12 });
      t.textContent = g.toFixed(1);
      svg.appendChild(t);
    });
    DMG.forEach(function (d, i) {
      var t = svgEl("text", { x: X(i), y: H - 24, "text-anchor": "middle",
        fill: "var(--frontBrightText)", "font-size": 12, "font-weight": "bold" });
      t.textContent = d.x;
      svg.appendChild(t);
    });
    var xl = svgEl("text", { x: ml + iw / 2, y: H - 5, "text-anchor": "middle",
      fill: "var(--frontBrightText)", "font-size": 10.5, "letter-spacing": "2",
      "font-weight": "bold" });
    xl.textContent = "CHANNELS REMOVED";
    svg.appendChild(xl);

    SER.forEach(function (s) {
      var d = "";
      DMG.forEach(function (row, i) { d += (i ? "L" : "M") + X(i) + " " + Y(row[s.k]); });
      svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: "var(" + s.v + ")",
        "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }));
      DMG.forEach(function (row, i) {
        svg.appendChild(svgEl("circle", { cx: X(i), cy: Y(row[s.k]), r: 4.5,
          fill: "var(" + s.v + ")", stroke: "var(--bg)", "stroke-width": 2 }));
      });
      // Direct label at the line's start, so identity is never colour-alone.
      var lab = svgEl("text", { x: ml + iw + 12, y: Y(DMG[0][s.k]) + 4,
        fill: "var(" + s.v + ")", "font-size": 12, "font-weight": "bold" });
      lab.textContent = s.name;
      svg.appendChild(lab);
    });

    DMG.forEach(function (row, i) {
      var hit = svgEl("rect", { x: X(i) - iw / (DMG.length * 2), y: mt,
        width: iw / DMG.length, height: ih, fill: "transparent" });
      hit.addEventListener("mousemove", function (e) {
        showTip("<strong>" + row.x + " of channels cut</strong><br>" + row.p + " params · " +
          row.macs + " arithmetic cut<br>overall " + row.all.toFixed(4) + "<br>small " +
          row.small.toFixed(4) + "<br>tiny " + row.tiny.toFixed(4), e.clientX, e.clientY);
      });
      hit.addEventListener("mouseleave", hideTip);
      svg.appendChild(hit);
    });
    host.appendChild(svg);

    var html = "<table><thead><tr><th>channels cut</th><th>params</th><th>arithmetic cut</th>" +
      "<th>mAP50</th><th>small</th><th>tiny</th></tr></thead><tbody>";
    DMG.forEach(function (r) {
      html += "<tr><td>" + r.x + "</td><td>" + r.p + "</td><td>" + r.macs + "</td>" +
        '<td class="' + (r.all === 0 ? "dead" : "hi") + '">' + r.all.toFixed(4) + "</td><td>" +
        r.small.toFixed(4) + "</td><td>" + r.tiny.toFixed(4) + "</td></tr>";
    });
    document.getElementById("dmg-table").innerHTML = html + "</tbody></table>";
  }
  drawDamage();

  /* 25% pruned, recovered, measured as compiled engines. */
  var RECM = [
    { name: "mAP50",           unp: 0.7776, one: 0.7297, it: 0.6771 },
    { name: "small plumes",    unp: 0.6061, one: 0.5516, it: 0.4386 },
    { name: "tiny plumes",     unp: 0.1376, one: 0.0960, it: 0.0653 },
    { name: "correct silence", unp: 0.9740, one: 0.9520, it: 0.9270 }
  ];
  /* Reference versus two treatments: the baseline stays neutral so only the
     two things being judged carry hue. */
  var ARMS = [
    { k: "unp", name: "unpruned",  v: "--cat-ref" },
    { k: "one", name: "one-shot",  v: "--cat-a" },
    { k: "it",  name: "iterative", v: "--cat-b" }
  ];
  function drawRecovery() {
    var host = document.getElementById("rec-chart");
    if (!host) return;
    var W = 780, H = 330, ml = 56, mr = 18, mt = 16, mb = 52;
    var iw = W - ml - mr, ih = H - mt - mb;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": "Neither pruned model beats the unpruned model on any accuracy measure" });
    var gw = iw / RECM.length, bw = (gw - 28) / 3;
    var Y = function (v) { return mt + ih - v * ih; };

    [0, 0.25, 0.5, 0.75, 1].forEach(function (g) {
      svg.appendChild(svgEl("line", { x1: ml, x2: ml + iw, y1: Y(g), y2: Y(g),
        stroke: "var(--hair)", "stroke-width": 1 }));
      var t = svgEl("text", { x: ml - 11, y: Y(g) + 4, "text-anchor": "end",
        fill: "var(--frontBrightText)", "font-size": 12 });
      t.textContent = g.toFixed(2);
      svg.appendChild(t);
    });

    RECM.forEach(function (m, gi) {
      ARMS.forEach(function (a, ai) {
        var x = ml + gi * gw + 14 + ai * bw;
        var y = Y(m[a.k]);
        // 2px surface gap between adjacent bars.
        var r = svgEl("rect", { x: x + 1, y: y, width: bw - 2, height: mt + ih - y,
          fill: "var(" + a.v + ")", rx: 3 });
        r.addEventListener("mousemove", function (e) {
          showTip("<strong>" + a.name + "</strong><br>" + m.name + " " + m[a.k].toFixed(4),
            e.clientX, e.clientY);
        });
        r.addEventListener("mouseleave", hideTip);
        svg.appendChild(r);
      });
      var t = svgEl("text", { x: ml + gi * gw + gw / 2, y: H - 30, "text-anchor": "middle",
        fill: "var(--front)", "font-size": 12, "font-weight": "bold" });
      t.textContent = m.name;
      svg.appendChild(t);
    });
    var note = svgEl("text", { x: ml + iw / 2, y: H - 8, "text-anchor": "middle",
      fill: "var(--frontBrightText)", "font-size": 10.5, "letter-spacing": "2",
      "font-weight": "bold" });
    note.textContent = "HIGHER IS BETTER · BOTH PRUNED ARMS LOSE ON ALL FOUR";
    svg.appendChild(note);
    host.appendChild(svg);

    var html = "<table><thead><tr><th>measure</th><th>unpruned</th><th>one-shot</th>" +
      "<th>iterative</th></tr></thead><tbody>";
    RECM.forEach(function (m) {
      html += "<tr><td>" + m.name + '</td><td class="hi">' + m.unp.toFixed(4) + "</td><td>" +
        m.one.toFixed(4) + "</td><td>" + m.it.toFixed(4) + "</td></tr>";
    });
    html += '<tr><td>parameters</td><td class="hi">7.03 M</td><td>4.24 M</td><td>4.51 M</td></tr>' +
      '<tr><td>throughput (img/s)</td><td class="hi">473.7</td><td>381.1</td><td>450.2</td></tr>' +
      '<tr><td>energy (J / 1000 frames)</td><td>52.1</td><td>54.3</td><td class="hi">46.5</td></tr>';
    document.getElementById("rec-table").innerHTML = html + "</tbody></table>";
  }
  drawRecovery();

  /* Chart / table toggles. The table view is not decoration: it is the
     accessibility fallback for the series whose contrast sits below 3:1. */
  [["dmg-toggle", "dmg-chart", "dmg-table"], ["rec-toggle", "rec-chart", "rec-table"]]
    .forEach(function (ids) {
      var b = document.getElementById(ids[0]),
          c = document.getElementById(ids[1]),
          t = document.getElementById(ids[2]);
      if (!b || !c || !t) return;
      b.addEventListener("click", function () {
        var on = b.getAttribute("aria-pressed") === "true";
        b.setAttribute("aria-pressed", String(!on));
        b.textContent = on ? "Show table" : "Show chart";
        c.hidden = !on;
        t.hidden = on;
      });
    });
})();
