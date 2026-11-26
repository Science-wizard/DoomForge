# Project brief

`brief.html` is the source of the 2-page PDF. It is generated with headless Chrome:

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=docs/brief/DoomForge_Brief.pdf \
  "file://$PWD/docs/brief/brief.html"
```

The PDF is the submitted artifact; the HTML is kept for reproducibility. Numbers in
the results table come from the diagnostic suite (`training/suite/`) on the held-out
test split; see `docs/RESULTS.md`.
