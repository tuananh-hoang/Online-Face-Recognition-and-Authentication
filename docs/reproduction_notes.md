# Reproduction Notes: Chou et al. 2019

This repository is the evaluation code for "Data-specific Adaptive Threshold for Face Recognition and Authentication" (arXiv:1810.11160). The original protocol is an online open-set registration/authentication simulation, not the standard static LFW pair protocol.

## 1. Dataset

The checked-in data currently includes precomputed feature CSVs for:

- `data/lfw/features_lfw_v0.csv` through `features_lfw_v9.csv`
- `data/color_FERET/features_color_FERET_v0.csv` through `features_color_FERET_v9.csv`

Each `v0`...`v9` file is one shuffled registration order. The matching `register_order_*.txt` files store the corresponding image order.

## 2. Main Experiment Scripts

Original adaptive-threshold experiment:

```bash
python simulator_v4_adaptive_thd.py data/color_FERET --max_compare_num 100
```

Original fixed-threshold experiment:

```bash
python simulator_v4_fixed_thd.py data/color_FERET 0.39
```

Average best accuracy across the 10 shuffled runs:

```bash
python get_avg_accuracy.py result/Simulator_v4_features_color_FERET_v
```

The same script pattern works for `data/lfw` if the corresponding threshold is supplied for the fixed-threshold simulator.

## 3. Embeddings

The original README states that embeddings were produced with FaceNet model `20170512-110547`. The current repo already includes the dumped embeddings in CSV form, so reproducing the simulator does not require TensorFlow or the FaceNet model unless new embeddings must be generated.

CSV rows have this logical format:

```text
image/person label, 128-d embedding vector, initial threshold, image path
```

`util.readEmb_csv()` loads the label and 128-dimensional vector. The image path is present in the CSV but ignored by the original simulator.

## 4. Threshold Computation

The adaptive threshold is implemented in `database.py`.

- `Database.insert()` appends the new embedding and calls `update_thresholds()`.
- `Database.update_thresholds()` compares the new embedding with previous embeddings from different identities.
- Each stored embedding keeps a threshold equal to the largest observed cross-identity similarity considered by the update routine.
- `max_compare_num` controls how many prior classes/images are sampled during threshold updates. If it is less than 1 in the simulator, the code passes the total dataset size and effectively compares all available prior records.

The fixed-threshold simulator uses one constant threshold for every trial.

## 5. Accuracy Computation

Accuracy is written by `util.show_and_save_v3()`. The simulator counts:

- `fa`: accepted query whose identity is not yet in the database.
- `fr`: rejected query whose identity is already in the database.
- `wa`: accepted query whose identity is in the database but whose top match is a different identity.

The reported original accuracy is:

```text
1 - (fa + fr + wa) / (accept + reject)
```

This is equivalent to Chou et al.'s online protocol accuracy, where true accept means accept with the correct top identity and true reject means reject a new identity.

## 6. Shuffle / Repeat

The paper and README use 10 random shuffles per dataset and report average accuracy. In this repo, those shuffles are represented by the `v0` through `v9` feature CSVs. The simulator processes every `features_*` CSV in the dataset directory.

## 7. Outputs

The original simulator writes result files under:

```text
result/Simulator_v4_features_<dataset>_v<repeat>
```

The result files contain `compare_num`, FAR, FRR, WAR, and accuracy lines. `get_avg_accuracy.py` scans these files and averages the best accuracy from each repeat.

## 8. Dependency Notes

The core simulators need only NumPy plus the local modules. The full environment in `requirements.txt` pins old packages (`numpy==1.19.0`, `scipy==1.5.1`, `matplotlib==3.2.2`) and the embedding dumper uses TensorFlow v1 compatibility plus OpenCV via `facenet_simple.py`. On modern Python versions, regenerating embeddings is likely harder than running the simulator from the precomputed CSVs.

An original adaptive-simulator smoke test was run on a 200-row LFW subset with the `face_edge` conda Python:

```bash
C:\Users\Hi\miniconda3\envs\face_edge\python.exe simulator_v4_adaptive_thd.py outputs/chou_reproduction/original_smoke_data --max_compare_num 20
```

After `tabulate`, `scipy`, and `matplotlib` were available in `face_edge`, the simulator itself ran. On Windows it first hit an output-path issue: `util.create_output_path()` builds a path like `result/Simulator_v4_original_smoke_data\features_lfw_v0` but only creates the top-level `result/` directory. Pre-creating `result/Simulator_v4_original_smoke_data` allowed the original smoke test to complete. The captured log is at `outputs/chou_reproduction/original_smoke.log`.

## Reproduction Scope For The New Experiment

The new experiment wrapper keeps the original FaceNet CSV embeddings and online registration order. It adds a held-out calibration split before the test split so fixed, M5, and M6 thresholds are not tuned on test trials. This means the new risk-constrained comparison is a fair extension on the same similarity scores, but it is not a byte-for-byte reproduction of the paper's fixed-threshold selection procedure, which used randomly generated verification pairs and 10-fold cross-validation.
