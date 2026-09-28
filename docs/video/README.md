# SQLsaber feature video

Source for the silent, looping product video on the [sqlsaber.com](https://sqlsaber.com) landing page
(`src/components/FeatureVideo.astro`).

The video is an HTML composition rendered frame by frame. Every frame is a pure function of time, so
renders are deterministic and edits are ordinary code changes.

| File | Purpose |
| --- | --- |
| `composition.html` | Stage, styles, and fonts (the site's own Fontsource files) |
| `composition.js` | Scenes and timeline (`T` holds each scene's start and end, in seconds) |
| `render.mjs` | Captures frames in headless Chromium and encodes them with ffmpeg |

## Scenes

1. **Intro**: the `[SQL]saber` lockup lights up as a dot-matrix display, then switches off.
2. **Ask**: the question from the README types out, then flies into the terminal.
3. **Demo**: `saber -d ./legislators.db "How many VPs became president by election in the 20th century?"`
   with the real tool steps, SQL, and result from the sample `legislators.db`.
4. **Safe by default**: `DROP TABLE legislators;` is struck through and rejected with the real SQL guard message.
5. **Stack and models**: supported databases and file formats, then providers.
6. **Workflow**: knowledge base, threads, Python SDK, MCP server.
7. **Outro**: the install command and sqlsaber.com.

Keep product claims in sync with the docs. The demo SQL (the `SQL` lines in `composition.js`) returns
the result shown in the video when run against the sample database in the repository root:

```bash
sqlite3 ../../legislators.db "SELECT e.name, t.start AS took_office
FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY executive_id ORDER BY start) AS n
      FROM executive_terms WHERE type = 'prez') t
JOIN executives e ON e.id = t.executive_id
WHERE t.n = 1 AND t.how = 'election'
  AND t.start BETWEEN '1901-01-01' AND '2000-12-31'
  AND e.id IN (SELECT executive_id FROM executive_terms WHERE type = 'viceprez');"
# Richard Nixon|1969-01-20
# George Bush|1989-01-20
```

The safety scene shows the message the SQL guard returns for `DROP TABLE legislators;`
(`validate_sql` in `src/sqlsaber/tools/sql_guard`).

## Preview

Serve `docs/` so the composition can load fonts from `docs/node_modules`:

```bash
cd docs
npm ci
npx http-server -p 8080 .    # or any static server rooted at docs/
# open http://localhost:8080/video/composition.html
```

Space pauses, the arrow keys seek by one second, and `?t=12.5` freezes a single frame.

## Render

Requires Node 22+, ffmpeg with libx264 and libaom-av1, and a Playwright Chromium.

```bash
cd docs/video
npm ci
npx playwright-core install chromium   # skip if Playwright's Chromium is already installed
npm run stills -- 2.4,12,18.5            # review PNGs in docs/video/stills/
npm run render                           # writes docs/public/video/
```

`npm run render` supersamples at 2x (3840x2160) and downscales, then writes:

- `sqlsaber-feature-av1.mp4`: AV1, preferred by browsers that support it
- `sqlsaber-feature.mp4`: H.264 fallback for older Safari and iOS
- `sqlsaber-feature-poster.jpg`: poster frame, also shown when the viewer prefers reduced motion

Set `FFMPEG` or `CHROMIUM_PATH` to use specific binaries. `--scale 1` renders faster for drafts.
