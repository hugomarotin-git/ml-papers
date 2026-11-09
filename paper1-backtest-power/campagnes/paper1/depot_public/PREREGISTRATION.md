# Pre-registered grids

Both grids below were written down on 2026-11-09 in the author's research notes **before any
result of the corresponding campaign was read**. They are reproduced here verbatim in
translation; the French originals are kept in the notes. The one deviation decided afterwards
is stated at the end, with its reason, as in Section 5.4 of the paper.

## Financial pipeline (campaigns A, B, C) — frozen 2026-11-09

- **Campaign A (power by cell).** Strengths **0.52 / 0.54 / 0.55 / 0.56 / 0.58 / 0.62 / 0.70**
  plus the null; 3 cadences (0.5 / 1 / 2 decisions per day) × 3 base risks (0.5 / 1 / 2 %);
  60 direction seeds (0–59); calendar seed 0; one path start every 3rd executed trade; AIMD
  sizing with the calibrated centres, no airbag. Verdict per cell: smallest detectable
  strength, or "uninterpretable" if the median censoring exceeds 10 % on the null side or on
  the signal side.
- **Campaign B (noise of the bar).** Same grid, strengths 0.55 and 0.58 only, null re-drawn
  with seeds 60–119 (B1) then 120–179 (B2). Measure: standard deviation of the null's p95
  over the three draws (A, B1, B2), per cell.
- **Campaign C (power as a function of n).** Cells 2/day × 1 % and 1/day × 2 %; strengths
  0.55 / 0.58 / 0.62; development window shortened to its most recent 1, 2 and 3 years, then
  the full window (4 years); 60 seeds.
- **Reading criterion, written in advance.** "Detectable" = median of the signal above the
  null's p95 AND interpretable cell; "clearly" if the margin exceeds the standard deviation of
  the bar measured in campaign B.

## Non-financial gate (campaigns HF) — frozen 2026-11-09

Dataset: "Hotel booking demand" (Antonio, de Almeida & Nunes, 2019, Data in Brief 22);
117,429 bookings after filtering (zero rate or zero nights removed), sorted by arrival date
(2015-07 → 2017-08); outcome = cancellation (37.5 %); stake = rate × nights winsorised at the
99th percentile (median 270). Stylised system: one decision per booking, correct → +stake,
wrong → −stake; signal planted by alignment (correct with probability = strength). Two-phase
gate (+10 % then +5 % of a 50,000 capital, maximum loss 10 % per phase) within a budget of N
decisions; sizing such that a decision of median stake risks r of the capital; lower bound =
passed / paths, censored counted as failures.

- **HF-A.** Budgets **125 / 250 / 500** decisions × risks **0.5 / 1 / 2 %**; strengths
  0.52 / 0.54 / 0.55 / 0.56 / 0.58 / 0.62 / 0.70 + null; seeds 0–59; one start every 100
  bookings (about 1,170 paths).
- **HF-A-unit.** Same grid, every stake replaced by the median (isolates the heavy tail).
- **HF-B1 / HF-B2.** Strengths 0.55 / 0.58, null re-drawn with seeds 60–119 then 120–179 →
  noise of the bar.
- **HF-C.** Cell 250 × 1 %, strengths 0.55 / 0.58 / 0.62, stream reduced to its most recent
  **125 / 250 / 500 / 770 / 2,000 / 5,000 / 20,000** usable starts (step 1 up to 2,000, then
  5 and 20 to keep about 1,000 paths).
- **Criteria identical to the financial case**: interpretable if censoring ≤ 10 % on both
  sides; detectable if p50 > null p95; "clearly" if the margin exceeds the standard deviation
  of the bar measured in HF-B.

## Deviation declared after the fact

- **Campaign C, 1-year window.** The 1-year point is structurally empty: 250 trading days of
  history with a 250-day budget leave a single start per seed, so the lower bound is 0 or 1 and
  the null's p95 is 1.000 by construction. The point was replaced by a 1.5-year window. The
  decision was taken after reading the 1-year run and **before** reading any result of the
  2- and 3-year runs.
- The nesting of the windows from the end of the development period (so that a longer window
  adds earlier data) was part of the grid; the resulting confound between sample size and
  market regime is reported in the paper as a limit of the design, not as a deviation.
