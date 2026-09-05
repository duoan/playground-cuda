// Book-wide style + reusable components for the Distributed Training book.
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

#let story(body) = block(
  fill: rgb("#fefce8"),
  stroke: (left: 3pt + rgb("#ca8a04")),
  inset: (x: 12pt, y: 8pt),
  radius: 2pt,
  width: 100%,
  [*面试故事.* #body],
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
//   Diagram helpers (built on CetZ) — distributed-training
// ==========================================================

// Ring layout for AllReduce visualization.
// n: number of ranks. labels: optional list of length n.
#let ring-diagram(n: 4, r: 1.6, labels: none, title: none,
                  node-color: rgb("#dbeafe"),
                  edge-color: rgb("#1e40af")) = {
  cetz.canvas({
    import cetz.draw: *
    let pi = 3.1415926535
    let pts = ()
    for i in range(n) {
      let a = pi / 2 - i * 2 * pi / n
      pts.push((r * calc.cos(a), r * calc.sin(a)))
    }
    // edges (arrows around the ring, clockwise from top)
    for i in range(n) {
      let (x1, y1) = pts.at(i)
      let (x2, y2) = pts.at(calc.rem(i + 1, n))
      // pull the arrow endpoints toward the center so they don't overlap
      let dx = x2 - x1; let dy = y2 - y1
      let L  = calc.sqrt(dx * dx + dy * dy)
      let ux = dx / L; let uy = dy / L
      let pad = 0.35
      line((x1 + ux * pad, y1 + uy * pad),
           (x2 - ux * pad, y2 - uy * pad),
           stroke: (paint: edge-color, thickness: 1pt),
           mark: (end: ">", size: 0.18))
    }
    for i in range(n) {
      let (x, y) = pts.at(i)
      circle((x, y), radius: 0.32, fill: node-color,
             stroke: 0.6pt + rgb("#1e3a8a"))
      let lab = if labels != none { labels.at(i) } else { "R" + str(i) }
      content((x, y), text(size: 10pt, weight: "bold", lab))
    }
    if title != none {
      content((0, r + 0.7),
              text(size: 10pt, weight: "bold", title))
    }
  })
}

// ---- Simple boxed FLOW diagram: horizontal boxes with arrows between them.
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

// ---- 1F1B / GPipe pipeline timeline visualization.
// stages: number of PP stages
// schedule: list of rows (one per stage); each row is a list of cells
//   ("F", n), ("B", n), ("W", n), ("_", n) with n = width in units.
#let pipeline-schedule(
  stages: 4,
  schedule: (),                 // list of (list of (kind, n))
  cell: 0.52,                   // widened default from 0.42 → 0.52
  gap-y: 0.12,
  title: none,
  legend: true,
) = {
  cetz.canvas({
    import cetz.draw: *
    let colors = (
      "F": rgb("#3b82f6"),   // forward - blue
      "B": rgb("#22c55e"),   // input-grad - green
      "W": rgb("#f97316"),   // weight-grad - orange
      "R": rgb("#ef4444"),   // recompute - red
      "_": rgb("#e5e7eb"),   // idle/bubble - gray
    )
    let kind-labels = ("F": "forward", "B": "input-grad",
                       "W": "weight-grad", "R": "recompute",
                       "_": "bubble")
    // Collect kinds actually present, in encounter order — avoids phantom "W"
    // in the legend when the schedule only uses F/B.
    let present = ()
    for row in schedule {
      for (kind, _) in row {
        if present.position(k => k == kind) == none {
          present.push(kind)
        }
      }
    }
    for s in range(schedule.len()) {
      let row = schedule.at(s)
      let y = -s * (cell + gap-y)
      // stage label at left
      content((-0.6, y - cell / 2),
              text(size: 8.5pt, weight: "bold", "S" + str(s)),
              anchor: "east")
      let x = 0
      for (kind, n) in row {
        let w = n * cell
        let col = colors.at(kind, default: rgb("#e5e7eb"))
        rect((x, y - cell), (x + w, y),
             fill: col, stroke: 0.3pt + rgb("#374151"))
        // Only draw the in-cell label when the cell is wide enough to hold
        // it — prevents overlapping smudges on dense 1F1B schedules.
        if kind != "_" and w >= 0.42 {
          content((x + w / 2, y - cell / 2),
                  text(size: 7.5pt, fill: white, weight: "bold", kind))
        }
        x = x + w
      }
    }
    if title != none {
      content((3, cell + 0.4),
              text(size: 10pt, weight: "bold", title))
    }
    if legend {
      let ly = -schedule.len() * (cell + gap-y) - 0.55
      let lx = 0
      for k in present {
        let lab = kind-labels.at(k, default: k)
        let col = colors.at(k, default: rgb("#cbd5e1"))
        rect((lx, ly - 0.3), (lx + 0.35, ly), fill: col,
             stroke: 0.3pt + rgb("#374151"))
        content((lx + 0.42, ly - 0.15),
                align(left + horizon,
                      text(size: 8pt, k + " = " + lab)),
                anchor: "west")
        // Space grows with label length.
        lx = lx + calc.max(1.8, lab.len() * 0.14 + 1.2)
      }
    }
  })
}

// ---- Memory breakdown horizontal stacked bar (for showing ZeRO / FSDP savings).
// entries: list of (label, value); shown as % of total.
#let mem-bar(entries, width: 8, bar-h: 0.5, gap: 0.15,
             palette: (rgb("#3b82f6"), rgb("#22c55e"),
                       rgb("#f97316"), rgb("#a855f7"),
                       rgb("#0ea5e9"), rgb("#ef4444"))) = {
  let total = 0.0
  for (_, v) in entries { total = total + v }
  cetz.canvas({
    import cetz.draw: *
    let i = 0
    for (label, val) in entries {
      let y = -i * (bar-h + gap)
      let w = width * val / total
      let col = palette.at(calc.rem(i, palette.len()))
      rect((0, y - bar-h), (w, y), fill: col,
           stroke: 0.4pt + rgb("#374151"))
      content((-0.2, y - bar-h / 2),
              align(right + horizon, text(size: 8pt, label)),
              anchor: "east")
      content((w + 0.15, y - bar-h / 2),
              align(left + horizon,
                    text(size: 8pt, weight: "bold",
                         str(calc.round(val * 10) / 10) + " GB")),
              anchor: "west")
      i = i + 1
    }
  })
}

// ---- Ring attention diagram: N ranks in a ring, each holds Q_i, K_i, V_i.
#let ring-attn-diagram(n: 4, r: 1.6, title: none) = {
  cetz.canvas({
    import cetz.draw: *
    let pi = 3.1415926535
    let pts = ()
    for i in range(n) {
      let a = pi / 2 - i * 2 * pi / n
      pts.push((r * calc.cos(a), r * calc.sin(a)))
    }
    for i in range(n) {
      let (x1, y1) = pts.at(i)
      let (x2, y2) = pts.at(calc.rem(i + 1, n))
      let dx = x2 - x1; let dy = y2 - y1
      let L = calc.sqrt(dx * dx + dy * dy)
      let ux = dx / L; let uy = dy / L
      let pad = 0.45
      line((x1 + ux * pad, y1 + uy * pad),
           (x2 - ux * pad, y2 - uy * pad),
           stroke: (paint: rgb("#dc2626"), thickness: 1.1pt, dash: "dashed"),
           mark: (end: ">", size: 0.18))
    }
    for i in range(n) {
      let (x, y) = pts.at(i)
      rect((x - 0.7, y - 0.42), (x + 0.7, y + 0.42),
           fill: rgb("#dbeafe"), stroke: 0.5pt + rgb("#1e3a8a"),
           radius: 3pt)
      content((x, y + 0.18),
              text(size: 9pt, weight: "bold", "GPU " + str(i)))
      content((x, y - 0.18),
              text(size: 8.5pt,
                   "Q" + str(i) + "·K" + str(i) + "·V" + str(i)))
    }
    content((0, 0),
            text(size: 8.5pt, weight: "bold",
                 fill: rgb("#dc2626"), "K/V rotate"))
    if title != none {
      content((0, r + 0.8),
              text(size: 10pt, weight: "bold", title))
    }
  })
}

// ==========================================================
//   Extended diagram helpers (added for figure/code expansion)
// ==========================================================

// ---- Collective diagram: N nodes in a row, each with 4 slots.
//   op: "AR" (AllReduce), "AG" (AllGather), "RS" (ReduceScatter), "A2A"
//   Shows the state BEFORE and AFTER the op side-by-side.
#let collective-diag(op: "AR", n: 4, slots: 4,
                     cell: 0.5, gap-x: 0.9,
                     colors: none) = {
  let palette = if colors != none { colors } else {
    (rgb("#3b82f6"), rgb("#22c55e"), rgb("#f97316"), rgb("#a855f7"),
     rgb("#ef4444"), rgb("#0ea5e9"), rgb("#eab308"), rgb("#ec4899"))
  }
  cetz.canvas({
    import cetz.draw: *
    // BEFORE side
    let draw-side(x0: 0, state: "before", title: "before") = {
      content((x0 + n * cell / 2, cell * (slots + 0.7)),
              text(size: 9pt, weight: "bold", title))
      for i in range(n) {
        for j in range(slots) {
          let fill = if state == "before" {
            // each rank has just its own colored slot
            if op == "AG" or op == "RS" {
              if j == i { palette.at(calc.rem(i, palette.len())) }
              else { rgb("#f3f4f6") }
            } else if op == "AR" {
              palette.at(calc.rem(i, palette.len())).lighten(30%)
            } else {  // A2A
              palette.at(calc.rem(j, palette.len())).lighten(20%)
            }
          } else {
            // after state
            if op == "AR" {
              rgb("#1f2937")   // reduced (dark)
            } else if op == "AG" {
              palette.at(calc.rem(j, palette.len())).lighten(10%)
            } else if op == "RS" {
              if j == 0 { palette.at(calc.rem(i, palette.len())) }
              else { rgb("#f3f4f6") }  // only slot 0 kept, and reduced
            } else {  // A2A
              palette.at(calc.rem(i, palette.len())).lighten(20%)
            }
          }
          rect((x0 + i * cell, j * cell),
               (x0 + (i + 1) * cell, (j + 1) * cell),
               fill: fill, stroke: 0.3pt + rgb("#374151"))
        }
        // rank label
        content((x0 + i * cell + cell / 2, -0.3),
                text(size: 8.5pt, weight: "bold", "R" + str(i)))
      }
    }
    draw-side(x0: 0, state: "before", title: "before")
    // arrow
    let x-arrow = n * cell + 0.25
    line((x-arrow, slots * cell / 2), (x-arrow + gap-x - 0.5, slots * cell / 2),
         stroke: 1pt + rgb("#111827"),
         mark: (end: ">", size: 0.2))
    content((x-arrow + (gap-x - 0.5) / 2, slots * cell / 2 + 0.25),
            text(size: 8pt, weight: "bold", op))
    draw-side(x0: n * cell + gap-x, state: "after", title: "after")
  })
}

// ---- Topology grid: rows × cols GPUs, each cell colored by a group id.
//   groups: 2D list matching shape rows × cols with integer group IDs.
//   group-labels: optional dict {id: "label"} for the legend.
#let topology-grid(rows: 2, cols: 4, groups: none, group-labels: none,
                   cell: 0.85, title: none,
                   palette: none) = {
  let pal = if palette != none { palette } else {
    (rgb("#dbeafe"), rgb("#dcfce7"), rgb("#fef3c7"),
     rgb("#fce7f3"), rgb("#e0f2fe"), rgb("#ede9fe"),
     rgb("#ffedd5"), rgb("#fee2e2"))
  }
  // If the maximum GPU id has 2+ digits, auto-widen the cell so labels
  // don't touch the borders.
  let max-id = rows * cols - 1
  let cell = if max-id >= 10 { calc.max(cell, 0.95) } else { cell }
  cetz.canvas({
    import cetz.draw: *
    for r in range(rows) {
      for c in range(cols) {
        let gid = if groups != none { groups.at(r).at(c) } else { 0 }
        let col = pal.at(calc.rem(gid, pal.len()))
        rect((c * cell, -(r + 1) * cell), (c * (cell) + cell, -r * cell),
             fill: col, stroke: 0.4pt + rgb("#374151"), radius: 2pt)
        content((c * cell + cell / 2, -r * cell - cell / 2),
                text(size: 8.5pt, weight: "bold",
                     "G" + str(r * cols + c)))
      }
    }
    if title != none {
      content((cols * cell / 2, 0.35),
              text(size: 9.5pt, weight: "bold", title))
    }
    if group-labels != none {
      let ly = -rows * cell - 0.5
      let lx = 0
      for (gid, lab) in group-labels {
        let col = pal.at(calc.rem(gid, pal.len()))
        rect((lx, ly - 0.3), (lx + 0.35, ly), fill: col,
             stroke: 0.3pt + rgb("#374151"))
        content((lx + 0.42, ly - 0.15),
                align(left + horizon, text(size: 7.5pt, lab)),
                anchor: "west")
        lx = lx + 1.7
      }
    }
  })
}

// ---- TP partition diagram: show weight W split column-wise or row-wise
//   across W ranks. mode: "column" or "row".
#let tp-partition(mode: "column", tp: 4, w: 3.6, h: 1.4,
                  title: none,
                  palette: (rgb("#dbeafe"), rgb("#dcfce7"),
                            rgb("#fef3c7"), rgb("#fce7f3"),
                            rgb("#e0f2fe"), rgb("#ede9fe"))) = {
  cetz.canvas({
    import cetz.draw: *
    if title != none {
      content((w / 2, h + 0.4),
              text(size: 9pt, weight: "bold", title))
    }
    if mode == "column" {
      // Split along the OUT dimension (columns of W: (out, in)).
      let strip = w / tp
      for i in range(tp) {
        let col = palette.at(calc.rem(i, palette.len()))
        rect((i * strip, 0), ((i + 1) * strip, h),
             fill: col, stroke: 0.4pt + rgb("#374151"))
        content((i * strip + strip / 2, h / 2),
                text(size: 10pt, weight: "bold", "R" + str(i)))
      }
      content((w / 2, -0.3),
              text(size: 8.5pt, "out dim (sharded across TP)"))
      content((-0.75, h / 2),
              text(size: 8.5pt, "in"))
    } else {
      // Split along the IN dimension (rows of W).
      let strip = h / tp
      for i in range(tp) {
        let col = palette.at(calc.rem(i, palette.len()))
        rect((0, i * strip), (w, (i + 1) * strip),
             fill: col, stroke: 0.4pt + rgb("#374151"))
        content((w / 2, i * strip + strip / 2),
                text(size: 10pt, weight: "bold", "R" + str(i)))
      }
      content((w / 2, -0.3),
              text(size: 8.5pt, "out dim"))
      content((-0.75, h / 2),
              text(size: 8.5pt, "in (shard)"))
    }
  })
}

// ---- SP+TP data-flow: sequence of tagged tensor shapes and ops.
//   steps: list of (shape-str, op-str-or-none)
// Op labels are drawn as clean text ABOVE each arrow (no badge) so they
// never overlap the arrow shaft or extend into adjacent boxes.
#let sp-tp-flow(steps: (),
                box-w: 2.8, box-h: 1.0, gap: 1.6) = {
  cetz.canvas({
    import cetz.draw: *
    let n = steps.len()
    for i in range(n) {
      let (shape, op) = steps.at(i)
      let x = i * (box-w + gap)
      rect((x, -box-h), (x + box-w, 0),
           fill: rgb("#eff6ff"), stroke: 0.5pt + rgb("#1e3a8a"),
           radius: 3pt)
      content((x + box-w / 2, -box-h / 2),
              text(size: 9pt, weight: "bold", raw(shape)))
      if i < n - 1 {
        line((x + box-w, -box-h / 2),
             (x + box-w + gap, -box-h / 2),
             stroke: 0.8pt + rgb("#374151"),
             mark: (end: ">", size: 0.18))
        if op != none {
          // Plain text label above arrow — enough room in gap now that
          // the default gap is 1.6.
          content((x + box-w + gap / 2, -box-h / 2 + 0.35),
                  text(size: 8.5pt, weight: "bold",
                       fill: rgb("#b45309"), op))
        }
      }
    }
  })
}

// ---- All-to-all diagram: bipartite lines between N sender columns and
// N receiver columns. Each sender has N pieces, each piece goes to a
// different receiver.
#let a2a-diag(n: 4, cell: 0.4, gap-y: 1.6, title: none) = {
  cetz.canvas({
    import cetz.draw: *
    if title != none {
      content((n * cell + (n - 1) * cell / 2, gap-y + n * cell + 0.5),
              text(size: 9pt, weight: "bold", title))
    }
    let palette = (rgb("#3b82f6"), rgb("#22c55e"), rgb("#f97316"),
                   rgb("#a855f7"), rgb("#ef4444"), rgb("#0ea5e9"))
    // Top row: senders. Each sender has n colored slots.
    for i in range(n) {
      for j in range(n) {
        let col = palette.at(calc.rem(j, palette.len()))
        rect((i * (n + 1) * cell + j * cell, gap-y),
             (i * (n + 1) * cell + (j + 1) * cell, gap-y + cell),
             fill: col, stroke: 0.3pt + rgb("#374151"))
      }
      content((i * (n + 1) * cell + n * cell / 2, gap-y + cell + 0.25),
              text(size: 8.5pt, weight: "bold", "R" + str(i) + " send"))
    }
    // Bottom row: receivers. Each has n slots to be filled with j-th slot
    // = piece from R_j.
    for i in range(n) {
      for j in range(n) {
        let col = palette.at(calc.rem(i, palette.len())).lighten(20%)
        rect((i * (n + 1) * cell + j * cell, 0),
             (i * (n + 1) * cell + (j + 1) * cell, cell),
             fill: col, stroke: 0.3pt + rgb("#374151"))
      }
      content((i * (n + 1) * cell + n * cell / 2, -0.3),
              text(size: 8.5pt, weight: "bold", "R" + str(i) + " recv"))
    }
    // Draw lines: sender i's slot j → receiver j's slot i
    for i in range(n) {
      for j in range(n) {
        let x1 = i * (n + 1) * cell + j * cell + cell / 2
        let y1 = gap-y
        let x2 = j * (n + 1) * cell + i * cell + cell / 2
        let y2 = cell
        line((x1, y1), (x2, y2),
             stroke: (paint: rgb("#6b7280"), thickness: 0.3pt))
      }
    }
  })
}

// ---- MoE dispatch diagram: tokens (top) → router → experts (bottom).
//   n-tokens: number of tokens.
//   n-experts: number of experts.
//   routing: list of length n-tokens, each is an expert id.
#let moe-dispatch(n-tokens: 6, n-experts: 4, routing: none,
                  cell: 0.5, gap: 1.4, title: none) = {
  cetz.canvas({
    import cetz.draw: *
    let route = if routing != none { routing }
                else { range(n-tokens).map(t => calc.rem(t, n-experts)) }
    let palette = (rgb("#3b82f6"), rgb("#22c55e"),
                   rgb("#f97316"), rgb("#a855f7"),
                   rgb("#0ea5e9"), rgb("#eab308"))
    if title != none {
      content((n-tokens * cell / 2, gap + cell + 0.5),
              text(size: 9pt, weight: "bold", title))
    }
    // Tokens
    for t in range(n-tokens) {
      let col = palette.at(calc.rem(route.at(t), palette.len()))
      rect((t * cell, gap), ((t + 1) * cell, gap + cell),
           fill: col.lighten(30%), stroke: 0.3pt + rgb("#374151"))
      content((t * cell + cell / 2, gap + cell / 2),
              text(size: 7pt, weight: "bold", "t" + str(t)))
    }
    // Experts row
    let exp-w = (n-tokens * cell) / n-experts
    for e in range(n-experts) {
      let col = palette.at(calc.rem(e, palette.len()))
      rect((e * exp-w, -cell), ((e + 1) * exp-w, 0),
           fill: col, stroke: 0.4pt + rgb("#374151"))
      content((e * exp-w + exp-w / 2, -cell / 2),
              text(size: 8pt, weight: "bold", fill: white,
                   "E" + str(e)))
    }
    // Dispatch lines
    for t in range(n-tokens) {
      let e = route.at(t)
      let col = palette.at(calc.rem(e, palette.len()))
      line((t * cell + cell / 2, gap),
           (e * exp-w + exp-w / 2, 0),
           stroke: (paint: col, thickness: 0.5pt))
    }
  })
}

// ---- Memory stacked bar (compare multiple configurations).
//   configs: list of (label, entries) where entries = list of (part, gb).
#let mem-stack(configs: (), width: 9, bar-h: 0.55, gap: 0.35,
               palette: (rgb("#3b82f6"), rgb("#22c55e"),
                         rgb("#f97316"), rgb("#a855f7"),
                         rgb("#ef4444"), rgb("#0ea5e9"),
                         rgb("#eab308"))) = {
  // Compute max total for scaling.
  let max-total = 0.0
  for (_, entries) in configs {
    let t = 0.0
    for (_, v) in entries { t = t + v }
    if t > max-total { max-total = t }
  }
  cetz.canvas({
    import cetz.draw: *
    // Legend: use the first config's entry labels
    let legend-labels = configs.at(0).at(1).map(e => e.at(0))
    let ly = 0.4
    let lx = 0.0
    for i in range(legend-labels.len()) {
      let col = palette.at(calc.rem(i, palette.len()))
      rect((lx, ly), (lx + 0.3, ly + 0.3), fill: col,
           stroke: 0.3pt + rgb("#374151"))
      content((lx + 0.38, ly + 0.15),
              align(left + horizon, text(size: 7.5pt, legend-labels.at(i))),
              anchor: "west")
      lx = lx + 2.0
    }
    // Bars
    for k in range(configs.len()) {
      let (label, entries) = configs.at(k)
      let y = -k * (bar-h + gap)
      let x = 0.0
      let total = 0.0
      for (_, v) in entries { total = total + v }
      for i in range(entries.len()) {
        let (_, v) = entries.at(i)
        let w = width * v / max-total
        let col = palette.at(calc.rem(i, palette.len()))
        rect((x, y - bar-h), (x + w, y), fill: col,
             stroke: 0.3pt + rgb("#374151"))
        x = x + w
      }
      content((-0.2, y - bar-h / 2),
              align(right + horizon,
                    text(size: 8pt, weight: "bold", label)),
              anchor: "east")
      content((x + 0.15, y - bar-h / 2),
              align(left + horizon,
                    text(size: 8pt,
                         str(calc.round(total * 10) / 10) + " GB")),
              anchor: "west")
    }
  })
}

// ---- Timeline visualization: multiple streams (rows), each is a list
// of colored segments (label, width). Great for overlap illustrations.
#let timeline(streams: (), unit: 0.4, bar-h: 0.5, gap-y: 0.15,
              title: none,
              colors: none) = {
  let cmap = if colors != none { colors } else {
    ("compute": rgb("#3b82f6"),
     "comm":    rgb("#f97316"),
     "bubble":  rgb("#e5e7eb"),
     "recomp":  rgb("#a855f7"),
     "wait":    rgb("#94a3b8"),
     "dp":      rgb("#22c55e"),
     "tp":      rgb("#ef4444"),
     "pp":      rgb("#eab308"))
  }
  cetz.canvas({
    import cetz.draw: *
    if title != none {
      let max-x = 0.0
      for (_, segs) in streams {
        let x = 0.0
        for (_, w) in segs { x = x + w * unit }
        if x > max-x { max-x = x }
      }
      content((max-x / 2, bar-h + 0.35),
              text(size: 9.5pt, weight: "bold", title))
    }
    for i in range(streams.len()) {
      let (label, segs) = streams.at(i)
      let y = -i * (bar-h + gap-y)
      content((-0.25, y - bar-h / 2),
              align(right + horizon,
                    text(size: 8pt, weight: "bold", label)),
              anchor: "east")
      let x = 0.0
      for (kind, w) in segs {
        let wpx = w * unit
        let col = cmap.at(kind, default: rgb("#cbd5e1"))
        rect((x, y - bar-h), (x + wpx, y), fill: col,
             stroke: 0.25pt + rgb("#374151"))
        // Only draw the in-segment label when the segment is wide enough
        // to hold it comfortably — narrow segments become swatches only.
        // A single glyph at 7.5pt needs ~0.16 canvas units of width.
        let need-w = kind.len() * 0.16 + 0.25
        if wpx > need-w {
          let txt-color = if kind == "bubble" { rgb("#374151") }
                          else { white }
          content((x + wpx / 2, y - bar-h / 2),
                  text(size: 7.5pt, weight: "bold", fill: txt-color, kind))
        }
        x = x + wpx
      }
    }
  })
}

// ---- Horizontal stacked bar with % labels (for step-time breakdown).
//   entries: list of (label, value); shown as a single stacked bar.
#let stacked-bar(entries: (), width: 10, bar-h: 0.6,
                 title: none,
                 palette: (rgb("#3b82f6"), rgb("#f97316"),
                           rgb("#e5e7eb"), rgb("#a855f7"),
                           rgb("#22c55e"), rgb("#ef4444"))) = {
  let total = 0.0
  for (_, v) in entries { total = total + v }
  cetz.canvas({
    import cetz.draw: *
    if title != none {
      content((width / 2, bar-h + 0.35),
              text(size: 9pt, weight: "bold", title))
    }
    let x = 0.0
    for i in range(entries.len()) {
      let (label, v) = entries.at(i)
      let w = width * v / total
      let col = palette.at(calc.rem(i, palette.len()))
      rect((x, 0), (x + w, bar-h), fill: col,
           stroke: 0.3pt + rgb("#374151"))
      if w > 0.6 {
        content((x + w / 2, bar-h / 2),
                text(size: 7.5pt, weight: "bold", fill: white,
                     str(calc.round(100.0 * v / total)) + "%"))
      }
      x = x + w
    }
    // Legend below — auto-wrap to a second row when it would overflow.
    let ly-base = -0.5
    let lx = 0.0
    let row = 0
    for i in range(entries.len()) {
      let (label, v) = entries.at(i)
      let col = palette.at(calc.rem(i, palette.len()))
      let entry-text = label + " (" + str(calc.round(v * 10) / 10) + ")"
      let entry-w = calc.max(2.5, entry-text.len() * 0.16 + 0.7)
      if lx + entry-w > width and lx > 0 {
        lx = 0.0
        row = row + 1
      }
      let ly = ly-base - row * 0.45
      rect((lx, ly - 0.28), (lx + 0.28, ly), fill: col,
           stroke: 0.3pt + rgb("#374151"))
      content((lx + 0.34, ly - 0.14),
              align(left + horizon,
                    text(size: 7.5pt, entry-text)),
              anchor: "west")
      lx = lx + entry-w
    }
  })
}

// ---- Formula callout, boxed centered
#let formula(body) = block(
  fill: rgb("#f8fafc"),
  stroke: 0.5pt + rgb("#334155"),
  inset: (x: 14pt, y: 10pt),
  radius: 3pt,
  width: 100%,
  align(center, body),
)

// ---- Cost table for parallel strategies.
//   rows: list of (strategy, per-layer-comm-bytes, sync-count, notes)
#let cost-table(header: ([策略], [每层通信量], [同步次数], [备注]),
                ..rows) = {
  let cells = ()
  for r in rows.pos() {
    for c in r { cells.push(c) }
  }
  table(
    columns: (auto, auto, auto, 1fr),
    stroke: 0.4pt + gray,
    inset: 6pt,
    align: (left, left, center, left),
    ..header,
    ..cells,
  )
}

// ==========================================================
//   Line-plot helpers (LR schedules, loss curves)
// ==========================================================

// ---- Multi-series line plot with axes and legend.
// Each series: (label, list of (x, y)).
// x-range and y-range auto-computed from data (with 5% padding).
// Legend is rendered BELOW the plot area to avoid collisions with series
// that reach near the top of the plot.
#let line-plot(
  series: (),
  width: 9, height: 4.5,
  x-label: "step", y-label: "value",
  title: none,
  colors: none,
  y-log: false,
  x-max: none,
  y-max: none,
  y-min: none,
  markers: false,
) = {
  let palette = if colors != none { colors } else {
    (rgb("#3b82f6"), rgb("#ef4444"), rgb("#22c55e"),
     rgb("#f97316"), rgb("#a855f7"), rgb("#0ea5e9"),
     rgb("#eab308"), rgb("#ec4899"))
  }
  // Compute ranges.
  let all-x = ()
  let all-y = ()
  for (_, pts) in series {
    for (x, y) in pts {
      all-x.push(x)
      all-y.push(if y-log { calc.log(calc.max(y, 1e-9)) } else { y })
    }
  }
  let xmin = calc.min(..all-x)
  let xmax = if x-max != none { x-max } else { calc.max(..all-x) }
  let ymin = if y-min != none {
    if y-log { calc.log(calc.max(y-min, 1e-9)) } else { y-min }
  } else { calc.min(..all-y) }
  let ymax = if y-max != none {
    if y-log { calc.log(calc.max(y-max, 1e-9)) } else { y-max }
  } else { calc.max(..all-y) }
  let ypad = (ymax - ymin) * 0.05
  ymin = ymin - ypad
  ymax = ymax + ypad
  let xr = xmax - xmin
  let yr = ymax - ymin
  if yr == 0 { yr = 1 }

  cetz.canvas({
    import cetz.draw: *
    // Title
    if title != none {
      content((width / 2, height + 0.5),
              text(size: 10pt, weight: "bold", title))
    }
    // Axes
    line((0, 0), (width, 0),
         stroke: 0.6pt + rgb("#374151"))
    line((0, 0), (0, height),
         stroke: 0.6pt + rgb("#374151"))
    // Y label
    content((-0.6, height / 2),
            text(size: 8pt, y-label + if y-log { " (log)" } else { "" }))
    // X label
    content((width / 2, -0.55),
            text(size: 8pt, x-label))
    // Ticks (5 on each axis)
    for i in range(6) {
      let x = i * width / 5
      let v = xmin + i * xr / 5
      line((x, 0), (x, -0.1), stroke: 0.4pt + rgb("#6b7280"))
      content((x, -0.28),
              text(size: 6.5pt, str(calc.round(v * 10) / 10)))
      let y = i * height / 5
      let vy = ymin + i * yr / 5
      line((0, y), (-0.1, y), stroke: 0.4pt + rgb("#6b7280"))
      let ylabel = if y-log {
        "10^" + str(calc.round(vy * 10) / 10)
      } else {
        str(calc.round(vy * 1000) / 1000)
      }
      content((-0.15, y),
              align(right + horizon, text(size: 6.5pt, ylabel)),
              anchor: "east")
    }
    // Series
    for si in range(series.len()) {
      let (lab, pts) = series.at(si)
      let col = palette.at(calc.rem(si, palette.len()))
      let prev = none
      for (x, y) in pts {
        let px = (x - xmin) / xr * width
        let ylog = if y-log { calc.log(calc.max(y, 1e-9)) } else { y }
        let py = (ylog - ymin) / yr * height
        if prev != none {
          let (ppx, ppy) = prev
          line((ppx, ppy), (px, py),
               stroke: (paint: col, thickness: 1.1pt))
        }
        if markers {
          circle((px, py), radius: 0.05, fill: col, stroke: none)
        }
        prev = (px, py)
      }
    }
    // Legend — BELOW the plot area, wraps to multiple rows if needed.
    // We drop the swatches down by 0.9 to clear the x-axis labels & label.
    let legend-y = -0.9
    let legend-x = 0.0
    let row = 0
    for si in range(series.len()) {
      let (lab, _) = series.at(si)
      let col = palette.at(calc.rem(si, palette.len()))
      // Approx label pixel width; wraps when we exceed plot width.
      let entry-w = calc.max(2.2, lab.len() * 0.16 + 1.0)
      if legend-x + entry-w > width and legend-x > 0 {
        legend-x = 0.0
        row = row + 1
      }
      let ly = legend-y - row * 0.45
      line((legend-x, ly), (legend-x + 0.4, ly),
           stroke: (paint: col, thickness: 1.4pt))
      content((legend-x + 0.48, ly),
              align(left + horizon, text(size: 7.5pt, lab)),
              anchor: "west")
      legend-x = legend-x + entry-w
    }
  })
}

// ---- Vertical stack of labeled operations, with a right-side annotation
// column (typical use: Megatron TP one-layer forward). Each step is a tuple:
//   (op-label, shape-str, tag) where tag ∈ {"full", "shard-h", "shard-s", "comm"}
// A colored bar on the right shows shape state; annotations sit outside.
#let op-stack(steps: (),
              width: 4.6, cell-h: 0.7, gap-y: 0.15,
              title: none) = {
  let tag-colors = (
    "full":    rgb("#dbeafe"),      // (B, S, H) full — blue
    "shard-h": rgb("#fef3c7"),      // (B, S, H/T) hidden sharded — yellow
    "shard-s": rgb("#dcfce7"),      // (B, S/T, H) seq sharded — green
    "comm":    rgb("#fce7f3"),      // collective op — pink
  )
  let tag-labels = (
    "full":    "full",
    "shard-h": "hidden shard",
    "shard-s": "seq shard",
    "comm":    "collective",
  )
  cetz.canvas({
    import cetz.draw: *
    if title != none {
      content((width / 2, cell-h + 0.3),
              text(size: 10pt, weight: "bold", title))
    }
    // Widen box relative to width so long op names don't overflow.
    // Split: 65% for the labeled op box, 35% for the shape annotation.
    let box-w = width * 0.65
    let n = steps.len()
    for i in range(n) {
      let (op, shape, tag) = steps.at(i)
      let y = -i * (cell-h + gap-y)
      let col = tag-colors.at(tag, default: rgb("#f3f4f6"))
      // main box (op label)
      rect((0, y - cell-h), (box-w, y),
           fill: col, stroke: 0.5pt + rgb("#374151"), radius: 3pt)
      content((box-w / 2, y - cell-h / 2),
              text(size: 9pt, weight: "bold", op))
      // right side: shape annotation
      content((box-w + 0.15, y - cell-h / 2),
              align(left + horizon,
                    text(size: 8.5pt, fill: rgb("#1f2937"),
                         raw(shape))),
              anchor: "west")
      // downward arrow to next
      if i < n - 1 {
        line((box-w / 2, y - cell-h),
             (box-w / 2, y - cell-h - gap-y),
             stroke: 0.6pt + rgb("#374151"),
             mark: (end: ">", size: 0.15))
      }
    }
    // Legend across the bottom
    let ly = -n * (cell-h + gap-y) - 0.5
    let lx = 0.0
    for (tag, col) in tag-colors {
      let lab = tag-labels.at(tag)
      rect((lx, ly - 0.3), (lx + 0.3, ly), fill: col,
           stroke: 0.3pt + rgb("#374151"))
      content((lx + 0.36, ly - 0.15),
              align(left + horizon, text(size: 7.5pt, lab)),
              anchor: "west")
      lx = lx + calc.max(1.7, lab.len() * 0.14 + 1.0)
    }
  })
}

// ---- Vertical annotation (dashed line) on a plot area.
// Useful to mark "restart" or "loss spike" events on training curves.
#let annotate-vline(x-frac: 0.5, height: 4.5, width: 9,
                    label: "", color: rgb("#dc2626")) = {
  cetz.canvas({
    import cetz.draw: *
    let x = x-frac * width
    line((x, 0), (x, height),
         stroke: (paint: color, thickness: 0.6pt, dash: "dashed"))
    if label != "" {
      content((x, height + 0.2),
              text(size: 7pt, fill: color, weight: "bold", label))
    }
  })
}
