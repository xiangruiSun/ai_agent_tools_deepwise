# Chest CT acquisition and analysis rules used by this system
source: project rules implemented in dwagent/ct/store.py and dwagent/ct/detector.py
audience: clinician

## Acquisition checks before automated analysis
Automated nodule analysis runs only when the series is CT, the slice thickness is 2.5 mm or less, there are no gaps between slices, and there are at least 60 slices. A series that fails any check is rejected and the study is routed to a rescan or manual-read path; no detector is run on it.

## Contrast-enhanced series
Contrast raises the density of vessels and soft tissue. Density-based typing of a nodule as solid or part-solid is less reliable on contrast-enhanced series, so the check raises a warning that is carried into the draft report.

## What the candidate detector can and cannot find
The detector in this project, classical_v1, is a rule-based research stand-in. It finds compact dense objects inside the lungs. It does not find pure ground-glass nodules, and it can miss nodules that are fused to vessels or to the chest wall. Each scan returns up to 12 ranked candidates, most of which are normal vessels. A radiologist must review every candidate; nothing the detector outputs is a diagnosis.

## Critical findings
A finding in category 4B or 4X is treated as a critical result. The pipeline flags it, and the report cannot be finalised until a radiologist has acknowledged the flag at sign-off.
