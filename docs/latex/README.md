# LaTeX sources — CT Dose AI report and presentation

| File | What it is |
|---|---|
| `CT_Dose_AI_Report.tex` | The report (article class, A4, 16 pages) |
| `CT_Dose_AI_Presentation.tex` | The presentation (Beamer, 16:9, 19 slides) |
| `CT_Dose_AI_Cheatsheet.tex` | The medical & physics cheatsheet (2-column, 3 pages) |
| `CT_Dose_AI_Advanced_Track.tex` | The advanced/deep-learning track: pre-scan model, virtual dose reduction, scout CNN |
| `architecture.tex` | The system-architecture diagram in TikZ, used by both documents |
| `make_figs.py` | Regenerates every chart in `figs/` from the trained model |
| `figs/` | All figures (PNG) |

## Build

Both documents compile with plain **pdfLaTeX** — no shell-escape, no external fonts.
Run the command twice so the page references settle:

```
pdflatex CT_Dose_AI_Report.tex
pdflatex CT_Dose_AI_Report.tex
pdflatex CT_Dose_AI_Presentation.tex
pdflatex CT_Dose_AI_Presentation.tex
pdflatex CT_Dose_AI_Cheatsheet.tex
pdflatex CT_Dose_AI_Cheatsheet.tex
pdflatex CT_Dose_AI_Advanced_Track.tex
pdflatex CT_Dose_AI_Advanced_Track.tex
```

The cheatsheet additionally uses `extarticle`, `multicol` and `textalpha` (the last one lets the Greek
glossary compile with plain pdfLaTeX).

On Windows: install MiKTeX (miktex.org) or TeX Live, then run the commands above in this folder,
or open the .tex file in TeXworks / Overleaf and press the compile button. MiKTeX installs any
missing package on first use.

Packages used: `helvet, amsmath, amssymb, geometry, fancyhdr, titlesec, enumitem, parskip,
graphicx, booktabs, tabularx, array, colortbl, multirow, xcolor, tcolorbox, tikz, caption,
hyperref` (report) and `beamer` plus the same set (slides).

## Regenerating the figures

From the project root, with the Python environment installed and the models trained:

```
python docs/latex/make_figs.py
```

It reads `models/CCTA.joblib` and writes the PNGs into `figs/`. The two simulation figures
(`f6_bo.png`, `f7_bandit.png`) re-run the phantom and bandit simulations, so their curves change
slightly between runs.

## Colours

`ctblue #2A78D6` · `ctbluedark #1A4F94` · `ctaqua #1BAF7A` · `ctorange #EB6834` — defined at the top
of each .tex file; change them there to restyle both documents.
