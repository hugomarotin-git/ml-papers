# MANIFEST — what goes into the public repository

Computed on 2026-11-09 from the import closure of the paper scripts (`driver.py`, `analyse.py`,
`analyse_n.py`, `figures.py`, `tables.py`, `hors_finance/*`) plus the data chain that produces
the panel the driver reads. Every path is relative to `machinelearning/`.

## Code — required (25 modules + data chain)

```
juge/aimd.py                juge/cadence.py            juge/calibration.py
juge/chemins.py             juge/controle_positif.py   juge/methodes_pfunded.py
juge/ordonnanceur.py        juge/regles_ftmo.py        juge/simulateur.py
juge/trajectoires.py        juge/twin_null.py
src/atr.py                  src/ftmo_cleaner.py        src/intraday_filter.py
src/label_triple_barriere.py  src/periodes.py          src/poids_echantillon.py
campagnes/paper1/driver.py  campagnes/paper1/analyse.py  campagnes/paper1/analyse_n.py
campagnes/paper1/figures.py campagnes/paper1/tables.py
campagnes/paper1/hors_finance/porte_sequentielle.py
campagnes/paper1/hors_finance/sanity_check_porte_sequentielle.py
campagnes/paper1/hors_finance/campagne_hf.py
campagnes/paper1/hors_finance/analyse_hf.py
```

## Proof suites — self-contained within this perimeter (15)

```
juge/sanity_check_aimd.py            juge/sanity_check_calibration.py
juge/sanity_check_controle_positif.py juge/sanity_check_methodes.py
juge/sanity_check_ordonnanceur.py    juge/sanity_check_pfunded.py
juge/sanity_check_regles_ftmo.py     juge/sanity_check_simulateur.py
juge/sanity_check_trajectoires.py    juge/sanity_check_twin_null.py
src/sanity_check_atr.py              src/sanity_check_ftmo_cleaner.py
src/sanity_check_intraday_filter.py  src/sanity_check_label.py
src/sanity_check_poids.py
```

`sanity_check_calibration`, `_methodes`, `_ordonnanceur` and `_trajectoires` import fixtures
from `sanity_check_simulateur.py`, which is included. **Run all fifteen once from the public
checkout before pushing**; a check that imports something outside the perimeter must be
dropped or the module added.

## NOT included (ML project, not needed by the paper)

`src/modele.py`, `features.py`, `matrice.py`, `cpcv.py`, `dsr.py`, `selection_i3.py`,
`compteur_essais.py`, `calibration_modele.py`, `portee.py`, `profil_horaire.py`,
`redondance.py`, `audit_lookahead.py`, `calendrier_news.py`, `vintage_compare.py`,
`dukascopy_confirmation.py`, `src/test_mutations.py` (tests the whole src, including excluded
modules), `juge/baseline.py`, `porte_du_juge.py`, `stabilite.py`, `table_null.py`,
`verdict_modele.py`, and their sanity checks. `notebooks/`, `CADRAGE.md`, the project README and
`Brain/` stay private.

## Outputs — all of `campagnes/paper1/sorties/` (1.5 MB)

```
2026-11-09T110741Z_repro-2026-09-15     reproduction of the September campaign (not cited)
2026-11-09T120632Z_A-puissance          + analyse_croisee.csv, fig1/fig2 (French)
2026-11-09T122734Z_B1-null-seeds60
2026-11-09T124919Z_B2-null-seeds120
2026-11-09T15*Z_C-{1j-2pct,2j-1pct}-{1,1.5,2,3}an   (the 1-year runs are kept: they document the deviation)
2026-11-09_C-puissance-n                puissance_n.csv, fig3 (French)
2026-11-09T1712*Z_HF-A, HF-A-unitaire, HF-B1-seeds60, HF-B2-seeds120, HF-C-n{125,...,20000}
2026-11-09_HF-analyse                   analyse_croisee_hf.csv, puissance_n_hf.csv, fig4/fig5 (French)
```

Logs of the runs (`j2.log`, `j3*.log`, `j4.log`), if kept, go to `campagnes/paper1/logs/`.

## Paper

The public README lives at the root of `paper1-backtest-power/`.

`campagnes/paper1/paper/`: `main.tex`, `related_work.tex`, `references.bib`, `tables/*.tex`
(generated), `figures/*.png` (generated, English), `main.pdf`. Exclude `main.log`, `main.blg`.

## Data — NOT in the repository

`data/` is outside the repository. The NASDAQ raw export and its derivatives are not
redistributed (proprietary feed). `data/hors_finance/hotels.csv` is downloaded by
`fetch_hotels.py` (sha256 7c2ae42a7353905ea136e5c2287f17c92c5435826598bfbb8491c6f0c7b1fc06).
`data/_HOLDOUT_SCELLE/` is never touched.

## Before the first push — checklist

1. Fill the repository URL and the contact in `paper/main.tex` (title footnote and
   Reproducibility paragraph), recompile, check the PDF.
2. Copy the files above into the new repository (new account, not the private ones);
   add `.gitignore` with `data/`, `.venv/`, `*.log`, `*.blg`, `__pycache__/`.
3. `pip install -r depot_public/requirements.txt`; run the fifteen proof suites; run
   `python campagnes/paper1/tables.py` and check the four tables are byte-identical.
4. Decide the licence lines in the README (MIT + CC BY 4.0 proposed).
5. Tag `v1-draft`; the Zenodo DOI (J7) points at that tag.
