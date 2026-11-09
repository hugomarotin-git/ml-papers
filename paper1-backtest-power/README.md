# Statistical Power of Backtest Validation Pipelines — code, outputs and paper

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24093891.svg)](https://doi.org/10.5281/zenodo.24093891)

Folder `paper1-backtest-power` of the `ml-papers` repository. Run every command below from this folder.

Companion code for *Statistical Power of Backtest Validation Pipelines: Planted-Signal
Controls Against False Negatives* (Hugo Marotin, preprint, November 2026, version 1.0).

Everything the paper reports is read from a dated output file in `campagnes/paper1/sorties/`,
produced by the code in this repository. The tables and figures of the paper are regenerated
from those files by scripts; no number is typed by hand.

## Layout

```
src/            data chain: cleaner, triple-barrier label, sample weights (NASDAQ panel)
juge/           the evaluation chain: FTMO rules, simulator, paths, scheduler, AIMD sizing,
                twin null, planted-signal control (controle_positif.py) and their proof suites
campagnes/paper1/
  driver.py                campaign driver (financial pipeline)
  analyse.py               cross-draw analysis: noise of the null bar, verdicts per cell
  analyse_n.py             power as a function of usable history (campaign C)
  figures.py               Figure 1 and Figure 2 (set PAPER1_EN=1 for English labels)
  tables.py                Tables 1–4 of the paper, from the CSV outputs; prints the key numbers
  hors_finance/            the hotel-booking sequential gate: module, proof suite, driver, analysis
  sorties/                 dated outputs: meta.json (grid, seeds, commit), cellules.json, points.csv
  paper/                   LaTeX source, references.bib, generated tables/ and figures/
  depot_public/            PREREGISTRATION.md, MANIFEST.md, fetch_hotels.py, requirements.txt
```

## Data

**NASDAQ panel (Sections 3–5).** The hourly bars come from a proprietary-trading-firm feed
(MetaTrader 5 history export) and are **not redistributed**. The panel the driver reads,
`data/v2/nas100_h1_pondere.csv`, is rebuilt from a raw export with:

```bash
python src/ftmo_cleaner.py --raw data/ftmo_h1/RAW_ftmo_h1_nas100.csv --out data/v2/nas100_h1_clean.csv
python src/label_triple_barriere.py --input data/v2/nas100_h1_clean.csv --out data/v2/nas100_h1_labelle.csv
python src/poids_echantillon.py --input data/v2/nas100_h1_labelle.csv --out data/v2/nas100_h1_pondere.csv
```

The development window is 2021-01-22 to 2024-12-31 (23,312 bars, 1,020 trading days, 17,177
labelled candidates). A later hold-out period exists and was never read for this work.

**Hotel bookings (Section 6).** Public dataset (Antonio, de Almeida & Nunes, 2019, *Data in
Brief* 22, doi 10.1016/j.dib.2018.11.126). Download and verify it with:

```bash
python campagnes/paper1/depot_public/fetch_hotels.py        # writes ../data/hors_finance/hotels.csv
```

The script checks the SHA-256 recorded when the campaigns ran and refuses a different file.

## Environment

Python 3.14, `numpy`, `pandas`, `matplotlib` (exact versions in `depot_public/requirements.txt`).
Paths are resolved relative to this folder; the data directory is its sibling `../data`, i.e. `ml-papers/data/`, which is git-ignored.

## Proof suites

Every module has a sanity-check file that fails loudly, and the planted-signal control has a
built-in negative control (strength 0.5 must never be "detectable"). Run them before anything:

```bash
for f in src/sanity_check_*.py juge/sanity_check_*.py campagnes/paper1/hors_finance/sanity_check_*.py; do
  python "$f" || echo "FAIL $f"
done
```

## Reproducing the campaigns

Each run writes `campagnes/paper1/sorties/<UTC timestamp>_<tag>/` with `meta.json` (full grid,
seeds, git commit), `cellules.json` and `points.csv`. Timings are for nine worker processes on a
laptop.

```bash
# Campaign A — power by cell (9 cells × 7 strengths + null × 60 seeds, about 57 min)
python campagnes/paper1/driver.py --tag A-puissance

# Campaigns B1, B2 — redraw the null (about 21 min each)
python campagnes/paper1/driver.py --tag B1-null-seeds60  --edges 0.55 0.58 --seed-offset 60
python campagnes/paper1/driver.py --tag B2-null-seeds120 --edges 0.55 0.58 --seed-offset 120

# Campaign C — power as a function of usable history (two cells, three shortened windows)
for an in 1.5 2 3; do
  python campagnes/paper1/driver.py --tag C-1j-2pct-${an}an --cadences 1 --risques 0.02 --edges 0.55 0.58 0.62 --annees $an
  python campagnes/paper1/driver.py --tag C-2j-1pct-${an}an --cadences 2 --risques 0.01 --edges 0.55 0.58 0.62 --annees $an
done

# Hotel gate — the whole set runs in under a minute
H=campagnes/paper1/hors_finance
python $H/campagne_hf.py --tag HF-A
python $H/campagne_hf.py --tag HF-A-unitaire --enjeux-unitaires
python $H/campagne_hf.py --tag HF-B1-seeds60  --edges 0.55 0.58 --seed-offset 60
python $H/campagne_hf.py --tag HF-B2-seeds120 --edges 0.55 0.58 --seed-offset 120
for n in 125 250 500 770 2000; do python $H/campagne_hf.py --tag HF-C-n$n --budgets 250 --risques 0.01 --edges 0.55 0.58 0.62 --pas 1  --n-utilisable $n; done
python $H/campagne_hf.py --tag HF-C-n5000  --budgets 250 --risques 0.01 --edges 0.55 0.58 0.62 --pas 5  --n-utilisable 5000
python $H/campagne_hf.py --tag HF-C-n20000 --budgets 250 --risques 0.01 --edges 0.55 0.58 0.62 --pas 20 --n-utilisable 20000
```

The pipeline is deterministic at equal seeds: a rerun reproduces the published `points.csv`
value for value.

## Analyses, tables and figures

```bash
S=campagnes/paper1/sorties
python campagnes/paper1/analyse.py   $S/<A-puissance> $S/<B1> $S/<B2>            # analyse_croisee.csv
python campagnes/paper1/analyse_n.py $S/2026-11-09_C-puissance-n $S/<C-*>/ $S/<A-puissance>
python campagnes/paper1/hors_finance/analyse_hf.py $S/2026-11-09_HF-analyse \
   --a $S/<HF-A> --unitaire $S/<HF-A-unitaire> --b $S/<HF-B1> $S/<HF-B2> --c $S/<HF-C-n*>
PAPER1_EN=1 python campagnes/paper1/figures.py $S/<A-puissance>                  # paper/figures/fig1, fig2
PAPER1_EN=1 python campagnes/paper1/analyse_n.py ...                              # paper/figures/fig3
PAPER1_EN=1 python campagnes/paper1/hors_finance/analyse_hf.py ...                # paper/figures/fig4, fig5
python campagnes/paper1/tables.py                                                 # paper/tables/*.tex + key numbers
cd campagnes/paper1/paper && tectonic main.tex                                    # main.pdf
```

## Pre-registration and provenance

The grids, seeds and reading criteria were frozen before each campaign ran; they are
reproduced in `depot_public/PREREGISTRATION.md` together with the one deviation and its
reason. The `commit` field in every `meta.json` refers to the private development history from
which this repository was extracted; the code published here is that commit's code for every
file listed in `depot_public/MANIFEST.md`, with comments and one descriptive label in `meta.json`
edited for publication; no executable line was changed.

## Licence and citation

Code: MIT. Paper and figures: CC BY 4.0. Please cite the paper and, for the code and outputs, the Zenodo archive
doi:10.5281/zenodo.24093891 (concept DOI, resolves to the latest version).
