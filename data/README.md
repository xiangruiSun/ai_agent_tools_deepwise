# Data

Nothing in this folder is committed except this file. Everything is downloaded by scripts.

| Dataset | What | Source | Licence | How it gets here |
|---|---|---|---|---|
| LIDC-IDRI subset | 10 chest CT studies, 1,306 DICOM images, 0.69 GB | The Cancer Imaging Archive | CC BY 3.0 | `python scripts/download_lidc.py` |
| LIDC annotations | Four-reader nodule annotations of all 1,018 LIDC scans | SQLite file inside the `pylidc` package | MIT (package) | same script |
| PMC-Patients (ReCDS) | 20,400 published patient case summaries used here | Hugging Face `zhengyun21/PMC-Patients-ReCDS` | CC BY-NC-SA 4.0, non-commercial | see `_ref/` steps in the README |

## Citations

- Armato SG III, et al. *Data From LIDC-IDRI*. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
- Armato SG III, et al. The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI). *Medical Physics* 38:915-931, 2011.
- Clark K, et al. The Cancer Imaging Archive (TCIA). *Journal of Digital Imaging* 26:1045-1057, 2013.
- Zhao Z, et al. PMC-Patients. *Scientific Data* 10, 909 (2023).

## Which studies and why

`download_lidc.py` picks studies from the annotation database, not by hand: two scans with no
annotated nodule, and two scans whose largest annotated nodule falls in each size band
(under 6 mm, 6 to 8, 8 to 15, 15 mm and over), limited to slice thickness of 2.5 mm or less and
at most 160 slices. The chosen list is written to `data/lidc/manifest.json`.

## What is not here

INSPECT (Stanford) is not used. Its licence requires the data to stay on an encrypted machine
and forbids sending it to a non-HIPAA API.
