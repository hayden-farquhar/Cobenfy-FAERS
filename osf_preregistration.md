# Pre-registration: Real-World Pharmacovigilance of Xanomeline-Trospium (Cobenfy) in the FDA Adverse Event Reporting System

**Registration date:** 2026-04-16
**Platform:** Open Science Framework (OSF)
**Template:** OSF Preregistration (formerly AsPredicted format)

---

## 1. Study Information

### Title
Real-World Pharmacovigilance of Xanomeline-Trospium (Cobenfy/KarXT) Using FDA FAERS Post-Marketing Data: A Disproportionality Analysis With Active-Comparator Assessment

### Authors
Hayden Farquhar MBBS MPHTM

### Description
This study applies standard pharmacovigilance disproportionality methods to FDA Adverse Event Reporting System (FAERS) data to characterise the real-world safety profile of xanomeline-trospium (Cobenfy), the first muscarinic receptor-targeting antipsychotic approved in over 30 years (FDA approval: September 26, 2024). We compare Cobenfy's adverse event reporting patterns against established D2-antagonist antipsychotics using active-comparator analyses to contextualise findings within the existing antipsychotic landscape.

### Research questions
**Primary:** Which adverse events show disproportionality signals for xanomeline-trospium in the FDA FAERS database during the first ~18 months of post-marketing use?

**Secondary:**
1. How does Cobenfy's adverse event reporting profile compare to olanzapine, risperidone, aripiprazole, quetiapine, lurasidone, and brexpiprazole using active-comparator disproportionality?
2. What is the time-to-onset profile for gastrointestinal, cardiovascular, and psychiatric events associated with Cobenfy?
3. Do disproportionality signals differ by CYP2D6 inhibitor co-medication status?

---

## 2. Hypotheses

All hypotheses are directional based on the known muscarinic (M1/M4 agonist) and anticholinergic (trospium) mechanism of action, and pre-approval clinical trial data (EMERGENT-1 through -5, ARISE).

### H1 — Gastrointestinal signals (primary hypothesis)
Gastrointestinal adverse events (nausea, vomiting, constipation, dyspepsia) will produce the strongest disproportionality signals for Cobenfy, consistent with muscarinic agonism as the dose-limiting toxicity observed in clinical trials.

### H2 — Cardiovascular signals
Tachycardia and hypertension will show detectable disproportionality signals, reflecting the cardiovascular effects of muscarinic receptor modulation observed at higher doses in EMERGENT trials.

### H3 — Metabolic advantage
Metabolic adverse events (weight increased, hyperglycaemia, diabetes mellitus, dyslipidaemia) will show either no disproportionality signal or significantly lower reporting rates compared to D2-antagonist comparators, consistent with the absence of D2/5-HT2C-mediated metabolic disruption.

### H4 — Extrapyramidal symptom advantage
Extrapyramidal symptoms (dystonia, akathisia, parkinsonism, tardive dyskinesia) will show no disproportionality signal for Cobenfy and significantly lower reporting rates compared to D2-antagonist comparators, consistent with the non-dopaminergic mechanism.

---

## 3. Design Plan

### Study type
Retrospective observational pharmacovigilance study (case/non-case disproportionality analysis).

### Study design
We will analyse spontaneous adverse event reports submitted to the FDA FAERS database from Q4 2024 (first quarter post-approval) through the most recent available quarter at time of analysis (expected Q1 2026). This is a signal-generation study; disproportionality does not establish causation.

### Reporting guideline
READUS-PV (REporting of A Disproportionality analysis Using individual Safety reports in PharmacoVigilance; Khouri et al., Drug Safety, 2024).

---

## 4. Data Source

### Database
FDA Adverse Event Reporting System (FAERS), quarterly ASCII data files, publicly available at: https://fis.fda.gov/extensions/FPD-QDE-FAERS/FPD-QDE-FAERS.html

### Time window
Q4 2024 through the most recent available quarter (anticipated Q1 2026, ~18 months post-approval).

### Data tables used
DEMO (demographics), DRUG (drug exposures), REAC (adverse reactions), OUTC (outcomes), INDI (indications), THER (therapy dates), RPSR (reporter source).

---

## 5. Drug Definitions

### Index drug
Xanomeline-trospium, identified in FAERS by any of the following drug name patterns (case-insensitive substring matching on drugname or prod_ai fields):
- XANOMELINE
- COBENFY
- KARXT
- KAR-XT

### Active comparators
| Drug | Brand names matched |
|------|-------------------|
| Olanzapine | OLANZAPINE, ZYPREXA, ZALASTA, LYBALVI |
| Risperidone | RISPERIDONE, RISPERDAL, PERSERIS, UZEDY |
| Aripiprazole | ARIPIPRAZOLE, ABILIFY, ARISTADA |
| Quetiapine | QUETIAPINE, SEROQUEL |
| Lurasidone | LURASIDONE, LATUDA |
| Brexpiprazole | BREXPIPRAZOLE, REXULTI |

### Case definition
A report is classified as a "case" for a given drug if the drug appears with role_cod = 'PS' (primary suspect) or 'SS' (secondary suspect).

### Reference set
For standard disproportionality: all other drug-event pairs in the FAERS database during the same time period. For active-comparator analysis: reports for the specified comparator drug.

---

## 6. Prespecified Preferred Terms

### Primary analysis
All MedDRA Preferred Terms (PTs) with ≥3 reports for Cobenfy will be analysed. No restriction on which PTs are tested.

### Active-comparator prespecified PTs
The following 32 PTs are prespecified for the active-comparator analysis:

**Gastrointestinal (n=5):** Nausea, Vomiting, Constipation, Dyspepsia, Diarrhoea

**Cardiovascular (n=3):** Tachycardia, Hypertension, Blood pressure increased

**Metabolic (n=6):** Weight increased, Hyperglycaemia, Diabetes mellitus, Metabolic syndrome, Blood glucose increased, Dyslipidaemia

**Extrapyramidal (n=6):** Dystonia, Akathisia, Parkinsonism, Tardive dyskinesia, Extrapyramidal disorder, Tremor

**Hormonal (n=3):** Hyperprolactinaemia, Galactorrhoea, Amenorrhoea

**Sedation (n=2):** Somnolence, Sedation

**Cardiac conduction (n=1):** Electrocardiogram QT prolonged

**Anticholinergic (n=3):** Urinary retention, Dry mouth, Vision blurred

**Other (n=3):** Insomnia, Headache, Dizziness

---

## 7. Analysis Plan

### 7.1 Deduplication
Reports will be deduplicated by caseid, retaining only the most recent caseversion per case.

### 7.2 Disproportionality methods
For each drug-PT pair with ≥3 reports, we will compute:

| Method | Metric | Signal threshold |
|--------|--------|-----------------|
| Proportional Reporting Ratio (PRR) | PRR with 95% CI and Yates-corrected χ² | PRR ≥ 2, χ² ≥ 4, n ≥ 3 |
| Reporting Odds Ratio (ROR) | ROR with 95% Wald CI | Lower 95% CI > 1 |
| Empirical Bayesian Geometric Mean (EBGM) | EBGM and EB05 via MGPS (DuMouchel 1999) | EB05 ≥ 2 |
| Information Component (IC/BCPNN) | IC with IC025 credibility interval (Bate et al. 1998) | IC025 > 0 |

### 7.3 Signal definition
A drug-PT pair is classified as a **signal** if ≥3 of the 4 methods above are positive simultaneously.

### 7.4 Multiple testing correction
Benjamini-Hochberg false discovery rate (FDR) correction at α = 0.05 applied to ROR p-values across all tested PTs.

### 7.5 Active-comparator analysis
For each of the 32 prespecified PTs and each of the 6 comparators (192 total tests), we compute an active-comparator ROR using only Cobenfy and comparator reports (not the full FAERS database). Multiple testing correction: Bonferroni (α = 0.05/192 = 0.00026).

### 7.6 Time-to-onset analysis
For signals with ≥20 reports and available therapy dates, we fit Weibull distributions to the time-to-onset (event_dt − start_dt). Events are categorised as:
- Early onset: <30 days
- Intermediate: 30–180 days
- Late onset: >180 days

Stratification by: age (<65 vs ≥65 years), sex, CYP2D6 inhibitor co-medication (bupropion, paroxetine, fluoxetine).

### 7.7 Sensitivity analyses
1. **Primary suspect only:** Restrict to role_cod = 'PS'
2. **US reports only:** Restrict to reporter_country = 'US'
3. **HCP reporters only:** Restrict to occp_cod in (MD, HP, OT, PH, RN)
4. **Weber effect:** Quarterly trend analysis of Cobenfy report counts and signal strength to assess stimulated reporting

---

## 8. Statistical Software

- Python 3.14 with DuckDB, pandas, numpy, scipy
- Disproportionality functions adapted from a previously validated implementation (cross-validated against the openEBGM R package; Pearson r > 0.9999, signal agreement > 99%)

---

## 9. Known Limitations (stated a priori)

1. **Spontaneous reporting limitations:** FAERS data are subject to underreporting, reporting bias, and Weber effect (stimulated reporting for novel drugs). Disproportionality does not establish causality.
2. **Channelling bias:** Cobenfy may be preferentially prescribed to patients who failed prior antipsychotics, potentially enriching for treatment-resistant or multi-morbid populations. Active-comparator analyses partially but not fully mitigate this.
3. **Small sample size:** If Cobenfy case counts are low (<100), signal detection power is limited. We will report this transparently and frame findings as hypothesis-generating.
4. **Drug name matching:** Despite comprehensive pattern matching, some reports may be missed due to misspellings or non-standard drug name entries.
5. **Absence of denominator data:** FAERS does not include prescription volume; reporting rates cannot be interpreted as incidence rates.

---

## 10. Data availability
FAERS data are publicly available. Analysis code will be deposited on GitHub upon manuscript acceptance.
