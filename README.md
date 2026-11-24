# **TRACE**: Tracking & Rendering for Automated Citrus Evaluation

This model trains a citrus quality grader **only on synthetic Blender renders**, then tests it zero-shot on real photographs from an industrial lemon quality-control dataset.

This is v0, a one-day proof of concept. Everything runs on a 2-core CPU.

![results](docs/results.png)

## Result

Task: flag mouldy fruit (reject) against clean fruit (accept). Test set: 2,270 real lemon images of 35 fruits, 264 of them mouldy. No real image was used for training, model selection or threshold tuning.

| | Value |
|---|---|
| ROC AUC, ensemble of 3 seeds | **0.82** (95% CI 0.76–0.87, bootstrap over fruit) |
| ROC AUC, individual seeds | 0.81 ± 0.04 (0.85, 0.81, 0.77) |
| At the default threshold (p = 0.5) | catches 89% of mouldy images, keeps 55% of clean ones (balanced accuracy 0.72) |
| Catching 90% of mould costs | 46% of clean lemons wrongly rejected |
| All rot (mould + gangrene) vs clean | AUC 0.72 |
| Accuracy on held-out *synthetic* images | 96–98% |

The gap between 96–98% on synthetic data and 0.82 AUC on real photos is the **sim-to-real gap**. This project exists to measure and close it. The real-data target is 95%.

## How it works

**1. Procedural generator** (`gen/citrus.py`, Blender 5.3 as a Python module, Cycles on CPU, about 2 s per image)
- **Varieties:** lemon (elongated, with pointed ends), orange and mandarin, with randomised size, shape, lumpiness, colour, peel pores, stem scar and style scar.
- **Defects** are drawn from parameters, so **every label is exact by construction**:
  - mould: patchy grey-green or white mycelium, matte and fuzzy, with shrivelling;
  - soft rot: brown, water-soaked;
  - minor marks (small blemishes, scars, green patches) on *both* grades, so "any mark" is not a shortcut the model can learn.
- **Randomised:** 1–3 area lights at 3500–7000 K, camera height, tilt and lens, end-on versus side views, and backgrounds (pure black, rubber and green conveyor belts, wood, cloth, white).

**2. Grader** (`train/grade.py`): an ImageNet-pretrained MobileNetV3-Large (timm), fine-tuned on 1,200 renders (half accept, half reject) for 6 epochs at 160 px, with strong colour, blur and geometry augmentation. The last epoch is kept, because there are no real labels to select with.

**3. Zero-shot test** (`eval/eval_lemons.py`, `eval/ensemble.py`): image labels come from the dataset's region annotations.
- mould = image has a `mould` region;
- clean = no `mould` or `gangrene` region (small blemishes and spots are allowed, as in Class I/II produce).

## What moved the number

| Change to the *generator* (no real labels involved) | Mould AUC on real |
|---|---|
| v1: 300 renders, smooth idealised peel | 0.70 |
| v2: 1,200 renders, plus stem scars, peel pores, end-on views, lumpier shapes and brighter lemons. The v1 model was rejecting real clean lemons seen end-on, because of the dark stem scar. | 0.80 ± 0.06 per seed |
| v3: the first defect patch always faces the camera. In v2, some "reject" renders had the defect on the hidden side, which is label noise. Synthetic accuracy rose from 82% to 97%. | **0.81 ± 0.04 per seed; ensemble 0.82** |

One thing tried and dropped: re-estimating BatchNorm statistics on the unlabelled real images (AdaBN). It improved calibration but lowered AUC to 0.66.

## Limitations (honest list)

- **Only 35 physical fruits**, each photographed many times, so the confidence intervals are wide. They are computed by resampling fruits, not images.
- **Seed variance is real** (AUC 0.77–0.85), and there is no real validation set to choose between seeds. That's why the reported result is the ensemble.
- **The false-reject rate is too high for a packhouse.** At 90% mould recall, 46% of clean lemons are rejected.
- **Some label noise in the "clean" set.** A few lemons labelled clean show a whitish residue that looks like early mould.
- **Image level only.** There's no detection, tracking or conveyor yet; each image holds one fruit on a black background.

## Next (the full TRACE plan)

1. Generator realism: finer mould micro-texture, wet rot, peel scale variation, and rendering on a GPU (Kaggle) for 10× more data.
2. **DEM** fruit-on-conveyor dynamics (Altair EDEM Student), feeding poses into Blender.
3. Multi-fruit belt **videos** with exact ground-truth tracks: a detector plus ByteTrack, a per-fruit grade vote across views, belt-speed estimation and pick-time prediction for a delta robot.
4. More real test domains: smartphone orange datasets and factory-line citrus video. This tests whether the method is **environment-agnostic** as well as product-agnostic.

## Reproduce

```bash
pip install -r requirements.txt
bash run_all.sh          # downloads data + weights, renders, trains 3 seeds, evaluates, draws the figure
```

Outputs from this run are in `results/`: per-seed metrics and training logs, ensemble metrics, and per-image scores.

## Credits

- Real test data: **Lemons quality control dataset**, Maciej Adamiak, SoftwareMill with Amplus (2020), MIT licence, DOI [10.5281/zenodo.3965568](https://doi.org/10.5281/zenodo.3965568). It is used for evaluation only and not redistributed here.
- Backbone weights: `mobilenetv3_large_100` from [timm](https://github.com/huggingface/pytorch-image-models).
- Rendering: [Blender](https://www.blender.org/) (`bpy`).

MIT licence. Puneet Mishra, 2026.
