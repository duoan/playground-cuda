// Book-wide style + reusable components for the MoE book.
//
// Usage in book.typ:
//   #import "template.typ": *
//   #show: book.with(title: "...", subtitle: "...", author: "...")

#import "@preview/cetz:0.3.4"

#let book(
  title: "Untitled",
  subtitle: "",
  author: "",
  body,
) = {
  set document(title: title, author: author)
  set page(
    paper: "a4",
    margin: (x: 2.2cm, y: 2.4cm),
    numbering: "1",
    number-align: center,
  )
  set text(
    font: ("DejaVu Serif", "Droid Sans Fallback"),
    size: 10.5pt,
    lang: "zh",
  )
  set par(justify: true, leading: 0.72em)
  set heading(numbering: "1.1.1")

  show heading.where(level: 1): it => {
    pagebreak(weak: true)
    v(1.5em)
    block(text(size: 22pt, weight: "bold", it.body))
    v(0.8em)
  }
  show heading.where(level: 2): it => {
    v(0.6em)
    block(text(size: 14pt, weight: "bold", it.body))
    v(0.2em)
  }
  show heading.where(level: 3): it => {
    v(0.4em)
    block(text(size: 11.5pt, weight: "bold", it.body))
  }

  show raw.where(block: true): it => block(
    fill: rgb("#f5f5f5"),
    inset: (x: 10pt, y: 8pt),
    radius: 3pt,
    width: 100%,
    text(font: ("DejaVu Sans Mono", "Droid Sans Fallback"), size: 9pt, it),
  )
  show raw.where(block: false): it => box(
    fill: rgb("#f0f0f0"),
    inset: (x: 3pt, y: 0pt),
    outset: (y: 2pt),
    radius: 2pt,
    text(font: ("DejaVu Sans Mono", "Droid Sans Fallback"), size: 9.5pt, it),
  )

  // ---------- title page ----------
  align(center)[
    #v(4cm)
    #text(size: 30pt, weight: "bold", title)
    #v(0.6cm)
    #if subtitle != "" { text(size: 16pt, subtitle) }
    #v(3cm)
    #text(size: 14pt, author)
    #v(0.4cm)
    #text(size: 10pt, datetime.today().display("[year]-[month]-[day]"))
  ]
  pagebreak()

  // ---------- toc ----------
  outline(title: [目录], depth: 2, indent: auto)
  pagebreak()

  body
}

// ---- reusable callouts ----

#let note(body) = block(
  fill: rgb("#eef6ff"),
  stroke: (left: 3pt + rgb("#3b82f6")),
  inset: (x: 12pt, y: 8pt),
  radius: 2pt,
  width: 100%,
  [*Note.* #body],
)

#let warn(body) = block(
  fill: rgb("#fff7ed"),
  stroke: (left: 3pt + rgb("#f59e0b")),
  inset: (x: 12pt, y: 8pt),
  radius: 2pt,
  width: 100%,
  [*Warning.* #body],
)

#let insight(body) = block(
  fill: rgb("#f0fdf4"),
  stroke: (left: 3pt + rgb("#22c55e")),
  inset: (x: 12pt, y: 8pt),
  radius: 2pt,
  width: 100%,
  [*Key insight.* #body],
)

#let interview(body) = block(
  fill: rgb("#fdf4ff"),
  stroke: (left: 3pt + rgb("#a855f7")),
  inset: (x: 12pt, y: 8pt),
  radius: 2pt,
  width: 100%,
  [*面试考点.* #body],
)

// ---- ladder table ----
#let ladder(..rows) = {
  let header = ([*版本*], [*核心思路*], [*相对成本 / 收益*])
  let cells = ()
  for r in rows.pos() {
    cells += (r.at(0), r.at(1), r.at(2))
  }
  table(
    columns: (auto, 1fr, auto),
    stroke: 0.5pt + gray,
    inset: 6pt,
    align: (left, left, right),
    ..header,
    ..cells,
  )
}

// ==========================================================
//   Diagram helpers (built on CetZ) — MoE specific
// ==========================================================

// Draw a probability heatmap of shape (rows, cols).
// Each cell colored by probs[r][c] (assumed in [0, 1]).
// If top-k is set, draw a red border on the top-k largest cells in each row.
//
// Usage:
//   #prob-heatmap(
//     probs: ((.10, .60, .05, .25),
//             (.40, .10, .45, .05)),
//     row-labels: ("t0", "t1"),
//     col-labels: ("E0", "E1", "E2", "E3"),
//     topk: 2,
//     title: "gate_probs (softmax)",
//   )
#let prob-heatmap(
  probs: ((),),
  row-labels: (),
  col-labels: (),
  topk: 0,
  cell: 0.9,
  gap: 0.05,
  title: none,
  caption: none,
  base-color: rgb("#3b82f6"),
  topk-stroke: rgb("#dc2626"),
) = {
  cetz.canvas({
    import cetz.draw: *
    let rows = probs.len()
    let cols = if rows > 0 { probs.at(0).len() } else { 0 }
    let s = cell + gap

    // find top-k indices per row
    let topk-marks = ()
    for r in range(rows) {
      let row = probs.at(r)
      // sort indices by value descending
      let idxs = range(cols)
      // simple selection: build list of (val, idx) and pick largest topk
      let picks = ()
      let taken = ()
      for _ in range(topk) {
        let best-i = -1
        let best-v = -1e9
        for i in range(cols) {
          if not taken.contains(i) {
            let v = row.at(i)
            if v > best-v { best-v = v; best-i = i }
          }
        }
        if best-i >= 0 {
          taken.push(best-i)
          picks.push(best-i)
        }
      }
      topk-marks.push(picks)
    }

    for r in range(rows) {
      let y = -r * s
      for c in range(cols) {
        let x = c * s
        let p = probs.at(r).at(c)
        // fade base color by p (higher p = darker)
        let light-pct = calc.max(0, calc.min(100, int((1.0 - p) * 100)))
        let fill-c = base-color.lighten(light-pct * 1%)
        rect((x, y), (x + cell, y + cell),
             fill: fill-c, stroke: 0.4pt + rgb("#374151"))
        content((x + cell / 2, y + cell / 2),
                text(size: 9pt, fill: if p > 0.5 { white } else { black },
                     str(calc.round(p * 100) / 100)))
        if topk > 0 and topk-marks.at(r).contains(c) {
          rect((x - 0.02, y - 0.02), (x + cell + 0.02, y + cell + 0.02),
               stroke: 1.6pt + topk-stroke, fill: none)
        }
      }
      if row-labels.len() > r {
        content((-0.15, y + cell / 2),
                align(right + horizon, text(size: 9pt, weight: "bold",
                                            row-labels.at(r))),
                anchor: "east")
      }
    }
    if col-labels.len() == cols {
      for c in range(cols) {
        content((c * s + cell / 2, cell + 0.35),
                text(size: 9pt, weight: "bold", col-labels.at(c)))
      }
    }
    if title != none {
      content(((cols * s) / 2, cell + 1.0),
              text(size: 10pt, weight: "bold", title))
    }
    if caption != none {
      content(((cols * s) / 2, -rows * s - 0.4),
              text(size: 8.5pt, caption))
    }
  })
}

// Draw a horizontal bar chart, e.g. of expert load (M_e per expert).
// entries: list of (label, count).
// - label-w: horizontal space reserved for labels (chart shifts right by this).
// - Bars are drawn starting at x = 0; labels sit at x in [-label-w, -0.15].
//   The value annotation sits at the right end of each bar.
#let expert-load(entries, max: none, unit: "tokens", width: 8,
                 capacity: none, label-w: 2.0, bar-h: 0.35, gap: 0.15,
                 bar-color: rgb("#3b82f6"),
                 over-color: rgb("#dc2626")) = {
  let m = if max == none {
    calc.max(..entries.map(e => e.at(1)))
  } else { max }
  cetz.canvas({
    import cetz.draw: *
    let i = 0
    for (label, val) in entries {
      let y = -i * (bar-h + gap)
      let w = width * val / m
      let fill-c = if capacity != none and val > capacity {
        over-color
      } else { bar-color }
      // bar
      rect((0, y - bar-h), (w, y), fill: fill-c,
           stroke: 0.4pt + rgb("#1e3a8a"))
      // label: sits inside the reserved gutter to the left of the bar
      content((-0.2, y - bar-h / 2),
              align(right + horizon, text(size: 8pt, label)),
              anchor: "east")
      // value annotation at right end of bar
      content((w + 0.15, y - bar-h / 2),
              align(left + horizon, text(size: 8pt, str(val) + " " + unit)),
              anchor: "west")
      i = i + 1
    }
    // capacity line
    if capacity != none {
      let cx = width * capacity / m
      line((cx, 0.15), (cx, -i * (bar-h + gap) + bar-h + 0.05),
           stroke: (paint: rgb("#dc2626"), thickness: 1.2pt, dash: "dashed"))
      content((cx, 0.4),
              text(size: 7.5pt, fill: rgb("#dc2626"), weight: "bold",
                   "capacity = " + str(capacity)))
    }
  })
}

// Time-share stacked horizontal bar for cost breakdown (each row = one stage,
// value = % of total).  Uses a distinct color so it is not confused with the
// expert-load bar chart above.
#let time-share-bar(entries, width: 10, bar-h: 0.55, gap: 0.18,
                    label-w: 3.5, unit: "%",
                    palette: (rgb("#3b82f6"), rgb("#22c55e"),
                              rgb("#f97316"), rgb("#a855f7"),
                              rgb("#0ea5e9"), rgb("#94a3b8"))) = {
  let total = 0.0
  for (_, v) in entries { total = total + v }
  cetz.canvas({
    import cetz.draw: *
    let i = 0
    for (label, val) in entries {
      let y = -i * (bar-h + gap)
      let w = width * val / total
      let fill-c = palette.at(calc.rem(i, palette.len()))
      rect((0, y - bar-h), (w, y), fill: fill-c,
           stroke: 0.4pt + rgb("#374151"))
      content((-0.2, y - bar-h / 2),
              align(right + horizon, text(size: 8.5pt, label)),
              anchor: "east")
      // percentage on top of bar (right end, or inside if too narrow)
      let pct-str = str(calc.round(val * 10) / 10) + " " + unit
      content((w + 0.15, y - bar-h / 2),
              align(left + horizon,
                    text(size: 8pt, weight: "bold", pct-str)),
              anchor: "west")
      i = i + 1
    }
  })
}

// Draw the tensor-shape pipeline for a MoE forward.
// stages: list of (name, shape-str, note).
//
// Automatic width: box-w defaults large enough for typical (name/shape) pairs;
// notes are placed to the right of the box with a fixed gutter so they will
// not overlap with the shape text — but the caller is still responsible for
// keeping notes short enough to fit within page margins.
#let shape-pipeline(stages: (), box-w: 5.0, box-h: 1.0, gap-y: 0.55,
                    note-gap: 0.4) = {
  cetz.canvas({
    import cetz.draw: *
    let n = stages.len()
    for i in range(n) {
      let y = -i * (box-h + gap-y)
      let (name, shape, note) = stages.at(i)
      rect((0, y - box-h), (box-w, y),
           fill: rgb("#eff6ff"), stroke: 0.6pt + rgb("#1e3a8a"),
           radius: 3pt)
      // name on the top half of the box
      content((box-w / 2, y - box-h * 0.32),
              text(size: 9pt, weight: "bold", name))
      // shape on the bottom half of the box
      content((box-w / 2, y - box-h * 0.70),
              text(size: 8pt, font: "DejaVu Sans Mono", shape))
      // note to the right of the box, single line
      if note != "" {
        content((box-w + note-gap, y - box-h / 2),
                align(left + horizon,
                      text(size: 8pt, fill: rgb("#6b7280"), note)),
                anchor: "west")
      }
      if i < n - 1 {
        line((box-w / 2, y - box-h), (box-w / 2, y - box-h - gap-y + 0.05),
             stroke: 0.8pt + rgb("#374151"),
             mark: (end: ">", size: 0.18))
      }
    }
  })
}

// Draw dispatch: N tokens (top row) routed to E experts (bottom row) via K
// arrows each.  routes: list of length N; each entry is a list of length K
// giving the expert index chosen.  weights (optional): matching probability
// used to modulate arrow thickness.
#let dispatch-diagram(
  routes: ((),),
  weights: none,
  n-experts: 4,
  token-cell: 0.55,
  expert-w: 1.1,
  expert-h: 0.7,
  layer-gap: 2.4,
  h-gap: 0.15,
  title: none,
  caption: none,
) = {
  cetz.canvas({
    import cetz.draw: *
    let n = routes.len()
    let s-tok = token-cell + h-gap

    // top row: tokens
    let token-y = layer-gap
    let token-total-w = n * s-tok - h-gap
    let token-offset = 0
    for i in range(n) {
      let x = token-offset + i * s-tok
      rect((x, token-y), (x + token-cell, token-y + token-cell),
           fill: rgb("#dbeafe"), stroke: 0.5pt + rgb("#1e3a8a"))
      content((x + token-cell / 2, token-y + token-cell / 2),
              text(size: 8pt, "t" + str(i)))
    }

    // bottom row: experts (centered under tokens)
    let expert-total-w = n-experts * (expert-w + 0.25) - 0.25
    let expert-offset = (token-total-w - expert-total-w) / 2
    let expert-y = 0
    for e in range(n-experts) {
      let x = expert-offset + e * (expert-w + 0.25)
      rect((x, expert-y), (x + expert-w, expert-y + expert-h),
           fill: rgb("#fef3c7"), stroke: 0.6pt + rgb("#92400e"))
      content((x + expert-w / 2, expert-y + expert-h / 2),
              text(size: 8.5pt, weight: "bold", "E" + str(e)))
    }

    // arrows
    for i in range(n) {
      let src-x = token-offset + i * s-tok + token-cell / 2
      let src-y = token-y
      let picks = routes.at(i)
      for k in range(picks.len()) {
        let e = picks.at(k)
        let dst-x = expert-offset + e * (expert-w + 0.25) + expert-w / 2
        let dst-y = expert-y + expert-h
        let thick = if weights != none {
          0.3pt + 1.6pt * weights.at(i).at(k)
        } else { 0.55pt }
        let color = if k == 0 { rgb("#1e40af") } else { rgb("#7c3aed") }
        line((src-x, src-y), (dst-x, dst-y),
             stroke: (paint: color, thickness: thick),
             mark: (end: ">", size: 0.13))
      }
    }

    if title != none {
      content((token-total-w / 2, token-y + token-cell + 0.4),
              text(size: 10pt, weight: "bold", title))
    }
    if caption != none {
      content((token-total-w / 2, expert-y - 0.5),
              text(size: 8pt, caption))
    }
  })
}

// Draw all-to-all communication across EP ranks.
// Before / after packed layout on each rank.
// ranks: number of EP ranks (= num experts if 1 expert / rank).
// tokens-per-rank: list of lists — before[r] = list of (expert_id, count).
#let a2a-diagram(
  n-ranks: 4,
  before: ((),),
  after: ((),),
  cell-w: 0.55,
  cell-h: 0.5,
  gap-y: 2.2,
  colors: none,
  title: none,
  before-label: "before all-to-all:",
  after-label: "after all-to-all:",
  arrow-label: "all-to-all",
) = {
  let c-palette = if colors != none {
    colors
  } else {
    (rgb("#fee2e2"), rgb("#dbeafe"), rgb("#dcfce7"), rgb("#fef3c7"),
     rgb("#f3e8ff"), rgb("#fce7f3"), rgb("#ccfbf1"), rgb("#fed7aa"))
  }
  cetz.canvas({
    import cetz.draw: *
    let rank-gap = 0.6
    // find max width to align both rows
    let max-tokens = 0
    for r in range(n-ranks) {
      let sum = 0
      for (_, cnt) in before.at(r) { sum = sum + cnt }
      if sum > max-tokens { max-tokens = sum }
      let sum2 = 0
      for (_, cnt) in after.at(r) { sum2 = sum2 + cnt }
      if sum2 > max-tokens { max-tokens = sum2 }
    }
    let rank-w = max-tokens * cell-w + 0.4

    // ---- BEFORE row (top) ----
    let top-y = gap-y
    for r in range(n-ranks) {
      let x0 = r * (rank-w + rank-gap)
      // rank label
      content((x0 - 0.2, top-y + cell-h / 2),
              align(right + horizon,
                    text(size: 8pt, weight: "bold", "rank " + str(r))),
              anchor: "east")
      let cursor = x0
      for (eid, cnt) in before.at(r) {
        let col = c-palette.at(calc.rem(eid, c-palette.len()))
        for _ in range(cnt) {
          rect((cursor, top-y), (cursor + cell-w - 0.05, top-y + cell-h),
               fill: col, stroke: 0.35pt + rgb("#374151"))
          content((cursor + (cell-w - 0.05) / 2, top-y + cell-h / 2),
                  text(size: 6.5pt, "E" + str(eid)))
          cursor = cursor + cell-w
        }
      }
    }
    content((-1.5, top-y + cell-h / 2 + 0.35),
            text(size: 9pt, weight: "bold", before-label))

    // ---- AFTER row (bottom) ----
    let bot-y = 0
    for r in range(n-ranks) {
      let x0 = r * (rank-w + rank-gap)
      content((x0 - 0.2, bot-y + cell-h / 2),
              align(right + horizon,
                    text(size: 8pt, weight: "bold", "rank " + str(r))),
              anchor: "east")
      let cursor = x0
      for (eid, cnt) in after.at(r) {
        let col = c-palette.at(calc.rem(eid, c-palette.len()))
        for _ in range(cnt) {
          rect((cursor, bot-y), (cursor + cell-w - 0.05, bot-y + cell-h),
               fill: col, stroke: 0.35pt + rgb("#374151"))
          content((cursor + (cell-w - 0.05) / 2, bot-y + cell-h / 2),
                  text(size: 6.5pt, "E" + str(eid)))
          cursor = cursor + cell-w
        }
      }
    }
    content((-1.5, bot-y + cell-h / 2 + 0.35),
            text(size: 9pt, weight: "bold", after-label))

    // vertical arrow in the middle
    let mid-x = (n-ranks * (rank-w + rank-gap)) / 2
    line((mid-x, top-y - 0.05), (mid-x, bot-y + cell-h + 0.05),
         stroke: (paint: rgb("#dc2626"), thickness: 1pt, dash: "dashed"),
         mark: (end: ">", size: 0.18))
    content((mid-x + 0.3, (top-y + bot-y) / 2 + cell-h / 2),
            align(left + horizon,
                  text(size: 8pt, fill: rgb("#dc2626"), arrow-label)),
            anchor: "west")

    if title != none {
      content((mid-x, top-y + cell-h + 0.9),
              text(size: 10pt, weight: "bold", title))
    }
  })
}

// Simple boxed FLOW diagram: horizontal boxes with arrows between them.
#let flow-boxes(
  boxes: (),
  box-w: 2.2,
  box-h: 0.9,
  gap-x: 0.5,
  colors: none,
) = {
  let palette = if colors != none {
    colors
  } else {
    (rgb("#eff6ff"), rgb("#f0fdf4"), rgb("#fef3c7"), rgb("#fce7f3"),
     rgb("#ecfeff"), rgb("#f5f3ff"))
  }
  cetz.canvas({
    import cetz.draw: *
    let n = boxes.len()
    for i in range(n) {
      let x = i * (box-w + gap-x)
      let col = palette.at(calc.rem(i, palette.len()))
      rect((x, -box-h), (x + box-w, 0),
           fill: col, stroke: 0.5pt + rgb("#374151"), radius: 3pt)
      content((x + box-w / 2, -box-h / 2),
              text(size: 9pt, weight: "bold", boxes.at(i)))
      if i < n - 1 {
        line((x + box-w, -box-h / 2),
             (x + box-w + gap-x, -box-h / 2),
             stroke: 0.7pt + rgb("#374151"),
             mark: (end: ">", size: 0.15))
      }
    }
  })
}
