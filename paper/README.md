# Paper

`main.tex` (IEEEtran, journal) + `refs.bib` -> `main.pdf`.

```bash
python figures/make_figures.py      # rebuild figures from ../results/<V>/<split>/metrics.json
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

On Overleaf: upload this folder (IEEEtran is built in).
