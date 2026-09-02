# Table paper-style spec (for subagents)

Purpose: every raw `#table(...)` in a chapter should be wrapped in a
`#figure(table(...), caption: [...], kind: table)` with a *paper-quality*
caption, and followed by a short *observation* paragraph.

## Reference example (already done)

Read `book/chapters/04_matmul.typ` around lines 454–500 for the launch-config
table, and lines 799–830 for the "各层职责" table.  Both show the required
pattern.

## Required transformation

For every raw `#table(...)` block in the assigned chapter:

1. **Wrap in figure**: replace `#table(...)` with

   ```typ
   #figure(
     table(   // ← note: NO leading # inside figure
       ... same body ...
     ),
     caption: [*Table:* <one-sentence description>. <expanded description>.
               <units and metric names when relevant>.],
     kind: table,
   )
   ```

   The `caption` MUST include:
   - a one-sentence description of *what* the table is
   - the units (μs, GB/s, %, etc.) if not already obvious in the column
   - the ncu metric name in backticks (e.g. `` `launch__grid_size` ``) when
     a column value comes from ncu, so a reader can reproduce
   - any caveat about the measurement (e.g. "log2N=27 to defeat L2",
     "only the first launch of a multi-stage kernel is shown")

2. **Add observation paragraph** immediately after (skip if the surrounding
   text already contains one — but check first, don't duplicate):

   ```typ
   *Observation*: <1-3 sentences pulling out the *pattern* in the numbers,
   not just restating them>. Prefer to link the pattern to the theory
   introduced earlier in the chapter.
   ```

   Good observations answer "what should the reader take away from this
   table that they wouldn't get by staring at the numbers?".

## Style rules

- Chinese for observations, matching chapter tone.
- No emoji.
- Use `*bold*` (single asterisks) not `**bold**`.
- Keep table body EXACTLY the same — only the wrapping changes.  Do NOT
  restructure columns.
- If a table already has a `#figure(...)` wrap, leave it alone.
- If a `#include "../bench/XX.typ"` is present, DO NOT touch it — the
  bench tables are already wrapped by `run_bench.py`.

## Typst gotchas

- Inside `#figure(...)`, `table(...)` must NOT have a leading `#`.
- In caption text, avoid raw `<something>` — typst parses `<...>` as a
  label.  If you need "<2%" write "小于 2%" or escape.
- Backticks in caption produce raw code style — that's what we want for
  metric names.
- `#insight[...]` / `#warn[...]` / `#note[...]` still work; use them for
  the observation if it's a key insight, otherwise a plain paragraph.

## Compile & verify

After edits:
```
cd /home/duo.an/workspaces/playground-cuda/book && ~/.local/bin/typst compile book.typ book.pdf
```
Must produce exit 0, zero errors.  Warnings only allowed if they existed
before your edits.

## Deliverable

- Number of tables wrapped.
- Number of observation paragraphs added (vs already existed).
- Any surprising thing you noticed about a table (e.g. a table whose
  numbers contradict text nearby).
- Confirm compile succeeded.

## Do NOT

- Do not rewrite prose that is already there — only *add* caption + observation.
- Do not change the chapter's structure or headings.
- Do not touch `template.typ`, `book.typ`, or bench `.typ` files.
- Do not invent metric names.
