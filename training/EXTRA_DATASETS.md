# Extra Datasets Registry

This document records all supplementary external datasets integrated to improve the detection accuracy of the two-wheeler traffic violation model, particularly focusing on underrepresented classes (`no_helmet` and `helmet`).

---

## Target Class Mapping Schema
All external datasets are strictly harmonized to the project's canonical class IDs:
- **`0`**: `helmet` (all wearing-helmet classes)
- **`1`**: `no_helmet` (names containing "without", "no helmet", "nohelmet", "not wearing", "head")
- **`2`**: `number_plate` (names containing "plate", "licen", "license plate", "registration")
- *All other classes are discarded during ingestion.*

---

## External Datasets Registry

| Dataset Name | Source / Universe URL | License | Original Classes | Mapped Classes | Added To Split |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Helmet and No Helmet Rider Detection** | [Roboflow Universe](https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection) | CC BY 4.0 | `with helmet`, `without helmet` | `0: helmet`, `1: no_helmet` | Train & Test |
| **Motorbike Helmet & Plate Safety** | [Roboflow Universe](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz) | CC BY 4.0 | `helmet`, `no helmet`, `plate` | `0: helmet`, `1: no_helmet`, `2: number_plate` | Train & Test |
| **Bike Helmet Detection** | [Roboflow Universe](https://universe.roboflow.com/jeev/bike-helmet-detection-2vdjo-uzil2) | CC BY 4.0 | `helmet`, `no-helmet` | `0: helmet`, `1: no_helmet` | Train & Test |

---

## Preprocessing & Validation Rules
1. **Polygon-to-Box Conversion**: Polygon annotations with $>4$ coordinate points are automatically bounded into tight bounding boxes $(x_{\text{center}}, y_{\text{center}}, w, h)$.
2. **Deduplication**: Exact image duplicates are detected and skipped using MD5 hash comparison against the existing dataset.
3. **Empty Filter**: Images without any surviving target annotations are discarded.
4. **Split Allocation**:
   - External `train` images $\to$ Ingested into `train`.
   - External `valid` / `test` images $\to$ Ingested into the new held-out `test` split (never touching `val`).

