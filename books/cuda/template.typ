// Book-wide style + reusable components.
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

// ---- kernel ladder table ----
// Usage:
//   #ladder(
//     ("naive",       "1 thread / element",     "~10%"),
//     ("grid-stride", "1 thread strides",       "~10%"),
//     ("vectorized",  "float4 load/store",      "~85%"),
//   )
// ============================================================
//   Diagram helpers (built on CetZ)
// ============================================================

// Draw an m × n grid of small cells, each optionally colored / labeled.
// active: array of (row, col) tuples that should be highlighted.
// cell-labels: dict "(r,c)" -> string
#let warp-grid(rows: 1, cols: 32, active: (), cell: 0.35, gap: 0.03,
               row-gap: 0.12, label-offset: 0.4, label-size: 8pt,
               colors: (active: rgb("#22c55e"), idle: rgb("#e5e7eb")),
               title: none, row-labels: (), col-labels: none) = {
  cetz.canvas({
    import cetz.draw: *
    let s = cell + gap
    // separate row stride: give each row its own vertical breathing room
    let row-stride = cell + row-gap
    let active-set = active.map(t => (t.at(0), t.at(1)))
    for r in range(rows) {
      for c in range(cols) {
        let x = c * s
        let y = -r * row-stride
        let is-active = active-set.contains((r, c))
        let fill = if is-active { colors.active } else { colors.idle }
        rect((x, y), (x + cell, y + cell), fill: fill,
             stroke: 0.4pt + rgb("#374151"))
      }
      if rows > 1 and row-labels.len() > r {
        // right-align the label so its text ends just before the first cell
        content((-label-offset, -r * row-stride + cell / 2),
                align(right + horizon, text(size: label-size, row-labels.at(r))),
                anchor: "east")
      }
    }
    if col-labels != none and col-labels != () {
      for c in range(cols) {
        content((c * s + cell / 2, cell + 0.3),
                text(size: 7pt, col-labels.at(c)))
      }
    }
    if title != none {
      content(((cols * s) / 2, -(rows - 1) * row-stride - cell - 0.4),
              text(size: 9pt, weight: "bold", title))
    }
  })
}

// Draw lane → memory-address mapping.  Renders:
//   top row:    N_lanes cells labeled 0..N_lanes-1 (the warp lanes that access)
//   bottom row: N_words cells (word addresses in a small range of memory)
//   arrows:     lane i -> word (mapping i)
// Highlights the words touched, marks 128B transaction boundaries (32 words),
// and reports how many transactions the pattern requires.
//
// Usage:
//   mem-access(mapping: (0, 1, 2, ..., 31), n-words: 32,
//              title: "coalesced: lane i -> word i")
//
// - mapping: length-N_lanes list of word indices (relative to the visible range).
//            entries outside [0, n-words) are drawn as arrows to off-canvas.
// - stride:  optional; if set and mapping is not given, auto-compute mapping[i] = i * stride.
// - highlight-txns: draw 128B (= 32 word) transaction boxes around each block of 32 words
//                   that contain at least one accessed word.
#let mem-access(mapping: none, stride: none, n-lanes: 32, n-words: 32,
                cell: 0.30, gap: 0.06, lane-row-y: 3.0, mem-row-y: 0.0,
                title: none, caption: none, highlight-txns: true,
                colors: (
                  active: rgb("#22c55e"),
                  touched: rgb("#fbbf24"),
                  idle: rgb("#e5e7eb"),
                  txn-hit: rgb("#dc2626"),
                  txn-none: rgb("#9ca3af"),
                )) = {
  let map = if mapping != none {
    mapping
  } else if stride != none {
    range(n-lanes).map(i => i * stride)
  } else {
    range(n-lanes)
  }

  cetz.canvas({
    import cetz.draw: *
    let s = cell + gap

    // ---- top row: lanes ----
    for i in range(n-lanes) {
      let x = i * s
      let is-active = i < map.len() and map.at(i) >= 0 and map.at(i) < n-words
      let fill = if is-active { colors.active } else { colors.idle }
      rect((x, lane-row-y), (x + cell, lane-row-y + cell),
           fill: fill, stroke: 0.35pt + rgb("#374151"))
      if calc.rem(i, 4) == 0 or i == n-lanes - 1 {
        content((x + cell / 2, lane-row-y + cell + 0.18),
                text(size: 6.5pt, "L" + str(i)))
      }
    }
    content((-0.5, lane-row-y + cell / 2),
            align(right, text(size: 8pt, weight: "bold", "warp")))

    // ---- bottom row: memory words ----
    let touched-set = ()
    for i in range(n-lanes) {
      if i < map.len() {
        let w = map.at(i)
        if w >= 0 and w < n-words {
          touched-set.push(w)
        }
      }
    }

    for w in range(n-words) {
      let x = w * s
      let is-touched = touched-set.contains(w)
      let fill = if is-touched { colors.touched } else { colors.idle }
      rect((x, mem-row-y), (x + cell, mem-row-y + cell),
           fill: fill, stroke: 0.35pt + rgb("#374151"))
      if calc.rem(w, 8) == 0 or w == n-words - 1 {
        content((x + cell / 2, mem-row-y - 0.18),
                text(size: 6.5pt, "w" + str(w)))
      }
    }
    content((-0.5, mem-row-y + cell / 2),
            align(right, text(size: 8pt, weight: "bold", "mem")))

    // ---- 128B transaction boundaries (every 32 words) ----
    if highlight-txns {
      let n-txns = calc.ceil(n-words / 32)
      let txn-hits = 0
      for t in range(n-txns) {
        let lo = t * 32
        let hi = calc.min((t + 1) * 32, n-words)
        // does this txn contain a touched word?
        let hit = false
        for w in range(lo, hi) {
          if touched-set.contains(w) { hit = true }
        }
        let color = if hit { colors.txn-hit } else { colors.txn-none }
        let stroke-w = if hit { 1.2pt } else { 0.4pt }
        rect((lo * s - 0.03, mem-row-y - 0.06),
             (lo * s + (hi - lo) * s - gap + 0.03, mem-row-y + cell + 0.06),
             stroke: (paint: color, thickness: stroke-w, dash: if hit { none } else { "dashed" }),
             fill: none)
        if hit { txn-hits = txn-hits + 1 }
      }
    }

    // ---- arrows lane -> memory ----
    for i in range(n-lanes) {
      if i < map.len() {
        let w = map.at(i)
        if w >= 0 and w < n-words {
          line((i * s + cell / 2, lane-row-y),
               (w * s + cell / 2, mem-row-y + cell),
               stroke: (paint: rgb("#3b82f6"), thickness: 0.35pt),
               mark: (end: ">", size: 0.10))
        }
      }
    }

    // title + caption
    if title != none {
      content(((n-lanes * s) / 2, lane-row-y + cell + 0.7),
              text(size: 9pt, weight: "bold", title))
    }
    if caption != none {
      content(((n-lanes * s) / 2, mem-row-y - 0.55),
              text(size: 8pt, caption))
    }
  })
}

// Draw lane -> memory-address mapping in the strided / scattered case:
//   segments: list of (segment_word_start, lane_id, lane_offset_in_segment)
// Each segment is drawn as a 4-word cell block (128 B / 4 words instead of 32
// to keep the picture readable at book width) with the accessed word
// highlighted; a "..." spacer separates segments; the arrow goes from lane_id
// (top row) to the highlighted word.
//
// Usage:
//   mem-access-scattered(
//     lane-words: (0, 32, 64, 96, 128, 160, 192, 224),
//     n-lanes: 8,
//     title: "uncoalesced: lane i -> word (i * 32)")
#let mem-access-scattered(lane-words: (), n-lanes: 8, cell: 0.35, gap: 0.06,
                          words-per-segment: 4, gap-between-segments: 0.7,
                          title: none, caption: none,
                          colors: (
                            active: rgb("#22c55e"),
                            touched: rgb("#fbbf24"),
                            idle: rgb("#e5e7eb"),
                            txn: rgb("#dc2626"),
                          )) = {
  cetz.canvas({
    import cetz.draw: *
    let s = cell + gap
    let n = lane-words.len()

    // segment_i occupies horizontal position:
    //   x = i * (words-per-segment * s + gap-between-segments)
    // Within segment_i, we draw `words-per-segment` cells, with the accessed
    // word placed in the middle (offset words-per-segment / 2).
    let seg-width = words-per-segment * s + gap-between-segments
    let hit-offset = int(words-per-segment / 2)  // where inside the segment the accessed word sits

    let total-width = n * seg-width - gap-between-segments
    let lane-y = 3.0
    let mem-y = 0.0

    // -- top row: lanes --
    for i in range(n) {
      let seg-x = i * seg-width
      let lane-x = seg-x + hit-offset * s
      rect((lane-x, lane-y), (lane-x + cell, lane-y + cell),
           fill: colors.active, stroke: 0.4pt + rgb("#374151"))
      content((lane-x + cell / 2, lane-y + cell + 0.22),
              text(size: 8pt, "L" + str(i)))
    }
    content((-0.5, lane-y + cell / 2),
            align(right, text(size: 8.5pt, weight: "bold", "warp")))

    // -- bottom row: memory segments --
    for i in range(n) {
      let seg-x = i * seg-width
      let word-addr = lane-words.at(i)
      // draw words-per-segment cells
      for j in range(words-per-segment) {
        let x = seg-x + j * s
        let is-hit = j == hit-offset
        let fill = if is-hit { colors.touched } else { colors.idle }
        rect((x, mem-y), (x + cell, mem-y + cell),
             fill: fill, stroke: 0.35pt + rgb("#374151"))
      }
      // 128B transaction box (surrounds all words-per-segment cells)
      rect((seg-x - 0.04, mem-y - 0.08),
           (seg-x + words-per-segment * s - gap + 0.04, mem-y + cell + 0.08),
           stroke: (paint: colors.txn, thickness: 1.4pt), fill: none)
      // address label
      content((seg-x + words-per-segment * s / 2 - gap / 2,
               mem-y - 0.35),
              text(size: 7pt, "addr " + str(word-addr * 4)))
      // arrow lane -> memory word
      let lane-x = seg-x + hit-offset * s
      line((lane-x + cell / 2, lane-y),
           (lane-x + cell / 2, mem-y + cell),
           stroke: (paint: rgb("#3b82f6"), thickness: 0.4pt),
           mark: (end: ">", size: 0.12))
      // "..." between segments
      if i < n - 1 {
        content((seg-x + words-per-segment * s - gap + gap-between-segments / 2,
                 mem-y + cell / 2),
                text(size: 10pt, weight: "bold", "..."))
      }
    }
    content((-0.5, mem-y + cell / 2),
            align(right, text(size: 8.5pt, weight: "bold", "mem")))

    if title != none {
      content((total-width / 2, lane-y + cell + 0.7),
              text(size: 9pt, weight: "bold", title))
    }
    if caption != none {
      content((total-width / 2, mem-y - 0.75),
              text(size: 8pt, caption))
    }
  })
}

// Show 32 lanes as a single row.  active can be a list of lane indices or a
// pattern like "even", "first-half", "if-then", "sequential-<n>", etc.
#let warp-lanes(active: (), cell: 0.32, title: none, note: none) = {
  let active-tuples = active.map(i => (0, i))
  align(center)[
    #warp-grid(rows: 1, cols: 32, cell: cell, active: active-tuples,
               row-labels: (), col-labels: ())
    #if note != none { text(size: 8pt, note) }
    #if title != none { linebreak() ; text(size: 9pt, weight: "bold", title) }
  ]
}

// Simple bar-chart helper (for perf comparisons that don't warrant a full ncu run)
#let hbar-chart(entries, max: none, unit: "", width: 8, height: none) = {
  let m = if max == none {
    calc.max(..entries.map(e => e.at(1)))
  } else { max }
  cetz.canvas({
    import cetz.draw: *
    let bar-h = 0.35
    let gap = 0.15
    let i = 0
    for (label, val) in entries {
      let y = -i * (bar-h + gap)
      let w = width * val / m
      rect((0, y - bar-h), (w, y), fill: rgb("#3b82f6"),
           stroke: 0.4pt + rgb("#1e3a8a"))
      content((-0.3, y - bar-h / 2), align(right, text(size: 8pt, label)))
      content((w + 0.15, y - bar-h / 2),
              align(left, text(size: 8pt, str(val) + " " + unit)))
      i = i + 1
    }
  })
}

// Tree-reduction diagram: draw the halving stages.
// stages: e.g. 4 means 16 -> 8 -> 4 -> 2 -> 1
// mode: "interleaved" (activity every other) or "sequential" (first half)
#let tree-reduction(mode: "sequential", n: 8, cell: 0.32) = {
  cetz.canvas({
    import cetz.draw: *
    let s = cell + 0.05
    let stages = calc.log(base: 2, n)
    let stages-int = int(stages)
    for stage in range(stages-int + 1) {
      let stride = calc.pow(2, stage)
      let y = -stage * (cell + 0.6)
      for i in range(n) {
        let is-active = if stage == 0 { true } else {
          if mode == "interleaved" {
            calc.rem(i, 2 * stride) == 0
          } else if mode == "sequential" {
            i < n / calc.pow(2, stage)
          } else { true }
        }
        let holds-value = if stage == 0 { true } else {
          if mode == "interleaved" { calc.rem(i, stride) == 0 }
          else if mode == "sequential" { i < n / calc.pow(2, stage - 1) }
          else { true }
        }
        let fill = if is-active { rgb("#22c55e") }
                   else if holds-value { rgb("#fef3c7") }
                   else { rgb("#f3f4f6") }
        rect((i * s, y - cell), (i * s + cell, y),
             fill: fill, stroke: 0.3pt + rgb("#374151"))
      }
      if stage > 0 {
        // draw arrows for adds
        for i in range(n) {
          let is-target = if mode == "interleaved" {
            calc.rem(i, 2 * stride) == 0
          } else if mode == "sequential" {
            i < n / calc.pow(2, stage)
          } else { false }
          if is-target {
            let src = if mode == "interleaved" { i + stride } else { i + n / calc.pow(2, stage) }
            if src < n {
              line((src * s + cell / 2, y + cell + 0.6 - cell),
                   (i * s + cell / 2, y),
                   stroke: 0.4pt + rgb("#374151"),
                   mark: (end: ">", size: 0.15))
            }
          }
        }
      }
      content((-0.3, y - cell / 2), align(right, text(size: 8pt, "stride " + str(stride))))
    }
  })
}

#let ladder(..rows) = {
  let header = ([*版本*], [*核心思路*], [*相对 peak 带宽*])
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
