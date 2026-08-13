# Pruning course — reusable page

An interactive course covering every pruning technique from MIT 6.5940 Lecture 3, grounded in
the wildfire-detection measurements. Written in the kernwerk.org house style.

```
pruning-course.html    the page (standalone, opens by double-click)
pruning-course.js      the three interactive widgets
```

No build step, no dependencies, no network calls. Open the HTML file directly in a browser and
it works.

## Two ways to use it

### 1. Standalone

Serve or open `pruning-course.html` as it is. It carries a local copy of the kernwerk tokens and
base rules, plus its own three-state theme toggle (light / night / automatic) that matches the
behaviour of the real site.

Both files must sit in the same directory, since the HTML loads the JS by relative path.

### 2. On kernwerk.org

The page is already written against your class names (`mainTitle`, `menu`, `sep`, `mono-tag`,
`lead`, `muted`, `accent`, `num`, `step`) and your `data-theme="light"` / `data-theme="night"`
contract, so integration is three deletions:

1. Delete `<style id="kernwerk-base">` and replace it with
   ```html
   <link rel="stylesheet" href="/style.css">
   <script src="/script.js"></script>
   ```
2. Delete `<script id="standalone-theme">` at the end of `<body>`. Your `script.js` owns the same
   cycle and writes the same `data-theme` values this page reads.
3. Swap the header and footer for your real ones, and point `.mainTitle` back at `/`.

Keep `<style id="course-components">`. It defines only the pieces that do not exist in
`style.css`: the chart colour tokens, the weight matrix, the channel bars, the panels and the
step cards.

Both blocks are marked with comments in the file saying exactly this.

## Notes on the design

**Fonts.** `@font-face` declares iA Writer Quattro with `local()` sources only, no URL. On a
machine with the font installed it renders identically to the site; everywhere else it falls back
to the system monospace stack. When hosting on kernwerk.org this is moot, since `style.css`
supplies the real `.woff2` files.

**Chart colours are not the brand colours, deliberately.** Two decisions worth keeping if you
edit them:

- The three plume size tiers are *nested subsets* (overall contains small contains tiny), which
  is ordinal rather than categorical data, so they use one hue stepped light to dark rather than
  three competing hues.
- The recovery chart is a reference against two treatments, so the unpruned baseline is neutral
  slate and only the two things being judged carry hue.

The brand red and yellow **fail colourblind separation when used as adjacent data series**
(ΔE 4.3 against a floor of 8, and no amount of darkening rescues that pair). The palettes shipped
here validate in both themes. Every chart also has a table view, which is the accessibility
fallback for the series sitting below 3:1 contrast, so the toggle is load-bearing rather than a
convenience.

**Accessibility.** Keyboard focus is visible, `prefers-reduced-motion` is respected, every chart
has an `aria-label` and a table equivalent, and series identity is always carried by a direct
label as well as by colour.

## Keeping it current

Experiment statuses live in the `.status` spans on each step card (`running`, `queued`,
`needs the board`, `optional`). Measured numbers live in two arrays at the bottom of
`pruning-course.js`: `DMG` for the damage curve and `RECM` for the recovery comparison. Nothing
else needs touching when results land.
