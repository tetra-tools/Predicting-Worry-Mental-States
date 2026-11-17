# Pre-specified phase protocol

Replication of Karim et al. 2017 (*In the grip of worry*, Transl Psychiatry 7:e1204) with the
LSTM attributions and the BOLD inputs. Implemented in [`phase_protocol.py`](phase_protocol.py);
run by `rerun_paper_analyses.sh` (step 2b) or:

```bash
python phase_protocol.py --pop test   # RAW external validation set, 88 participants: confirmatory
python phase_protocol.py --pop val    # FINA test set, 24 participants: replication (it also
                                      # selected the checkpoint, see readme)
```

## Why a protocol

The first reading of the network curves (two phases, a worry-specific second phase, matches
with the 2017 regions) was made after looking at the curves: the windows were read on them,
the networks were matched to the 2017 regions afterwards, and nothing was tested. It is a
hypothesis, not a result. This protocol fixes the windows, the regions, the measures and the
predicted directions before the analysis; the windows read on the curves are kept as
exploratory (family E).

## Windows (seconds of the figures, block onset = 0)

Task timing (`datasets/TimeSeriesDataset.py`): statement 0-9 s, worry / reappraisal period
10-24 s, fixation 25-34 s. With a 5 s hemodynamic delay:

| Window | Seconds | Content |
|---|---|---|
| baseline | 1-2 | BOLD of the preceding fixation, before the response to the block |
| induction | 6-15 | statement |
| maintenance | 16-24 | worry / reappraisal period |
| block | 6-24 | whole block (severity, as the 45 s blocks of 2017) |
| *induction_obs, maintenance_obs* | *7-11, 14-18* | *read on the curves: exploratory only* |

Second 25 is never used: it is the input of the explained output (prediction at the last
second), so it carries a large attribution by construction.

## Measures (per block, set and second; SHAP head = condition of the block)

| Measure | Definition | Question |
|---|---|---|
| amount | block-normalised \|SHAP\| (L2 norm over seconds 1-24 and the 481 ROIs) averaged over the set | share of the block's attribution (compositional) |
| **enrichment** | amount / median \|SHAP\| of the 481 ROIs at the same second | does the model use this set more than a typical ROI? |
| **direction** | Σ\|SHAP\|·sign(input − mean training block) / Σ\|SHAP\| | does the attribution come from activity above (+1) or below (−1) the average block? |
| **bold** | input z-scored per participant and ROI, minus the baseline window | activation vs the preceding fixation (2017: task vs rest) |

Sets: 15 anatomical sets (below) and the 18 network templates (positive part of each template).

## ROI sets (fixed before the analysis)

| Set | Rule | ROIs |
|---|---|---|
| visual_cortex | Schaefer `_Vis_` | 61 |
| thalamus | AAL3 `Thal_*` | 30 |
| caudate | AAL3 `Caudate_L/R` | 2 |
| sma_dacc | Schaefer `SalVentAttn_Med`, `Cont_Cing` | 19 |
| hippocampus, parahippocampal, amygdala | AAL3 | 2 each |
| posterior_insula | centroid \|x\| 32-44, y −28 to −2, z −6 to 16 | 8 |
| ba7_superior_parietal | \|x\| 12-36, y −75 to −45, z ≥ 45 | 17 |
| ba19_lateral_occipital | \|x\| ≥ 20, y ≤ −70, z 0-40 | 14 |
| mpfc | \|x\| ≤ 14, y ≥ 36, z −10 to 30 | 11 |
| ofc | \|x\| ≤ 45, y ≥ 15, z ≤ −8 | 23 |
| dlpfc | Schaefer `Cont_PFCl` | 23 |
| supramarginal | \|x\| ≥ 46, y −50 to −22, z 20-50 | 19 |
| inferior_parietal | \|x\| ≥ 36, y −75 to −50, z 20-50 | 13 |

Centroids: `external/schaefer400_7networks_centroids.csv` (MNI mm). Several 2017 regions
(caudate, hippocampus, parahippocampal gyrus, posterior insula, dlPFC, supramarginal and
inferior parietal cortex) carry ≤ 13 % of any network template, hence the anatomical sets.

## Hypotheses (from Karim et al. 2017)

| Family | Hypothesis | Measure | Window | Sets | Test |
|---|---|---|---|---|---|
| H1 initiation | worry activates | bold | induction | visual, thalamus, caudate, sma_dacc | Worry > 0 |
| H1 | worry-specific use | enrichment | induction | caudate, sma_dacc | Worry > Neutral |
| H1 | reappraisal lowers SMA/dACC | bold | induction | sma_dacc | Worry > Reappraisal |
| H1 | shared with reappraisal (no direction) | enrichment | induction | visual, thalamus | Worry vs Reappraisal |
| H2 maintenance | second-phase contrast | enrichment | maintenance | thalamus, hippocampus, parahippocampal, amygdala, posterior insula | Worry > Reappraisal, Worry > Neutral |
| H2 | worry activates | bold | maintenance | same | Worry > 0, Worry > Reappraisal |
| H2 | no habituation | bold | maintenance | same | slope across worry blocks > 0, and > neutral slope |
| H3 direction | reappraisal lowers medial frontal / OFC | bold, direction | induction | sma_dacc, mpfc, ofc | Reappraisal < 0 |
| H3 | worry attribution from increases | direction | induction | visual, thalamus, caudate, sma_dacc | Worry > 0 |
| H4 severity | activation | bold | block | BA19, BA7, sma_dacc, mpfc, ofc, dlpfc, supramarginal, inferior parietal | PSWQ slope > 0 (age, sex) |
| H4 | use by the model (no direction) | enrichment | block | same | PSWQ slope |
| H4 | after depression | bold | block | parahippocampal, hippocampus, amygdala, BA7 | PSWQ slope > 0 (age, sex, depression) |

Depression: HAMD (cohort 2) or MADRS (cohort 1). Block order within a condition comes from the
scan order of the blocks (the val and test loaders are not shuffled).

## Statistics

* Unit: the participant (blocks averaged within participant; direction = ratio of sums).
* Paired or one-sample t-tests in the predicted direction (two-sided where no direction is
  predicted); slopes per participant then one-sample or paired t-tests; OLS with
  standardised predictors for PSWQ.
* Benjamini-Hochberg FDR within each family (H1, H2, H3, H4, E). Verdict: *supported* if
  q < 0.05 in the predicted direction.
* Composition check (`vs_median_set`): estimate minus the median estimate of the 15
  anatomical sets. Amounts are shares, so one large effect shifts every set; enrichment
  reduces this (in a simulation, a planted thalamus effect of +840 % shifted the other sets by
  −6 to −8 % in enrichment, −41 % in amount), and a set-specific effect should stand well above
  the median of the sets.

## Reading the results

* **Replication of 2017** = H1-H4 on **bold** (activation), the counterpart of the 2017 CBF.
* **Contribution of the model** = enrichment and direction: whether the classifier relies on
  the same regions, in the same phase, through increases or decreases of activity.
* A second-phase contrast in enrichment without a bold effect means the model uses the
  pattern of the region (e.g. a decrease, or a difference from the other conditions), not
  that the region is more active.
* 2017 compared tasks with rest in 20 GAD and 20 controls (ASL, 45 s blocks); here the
  comparisons are between tasks or with the preceding fixation, in dimensional samples.
