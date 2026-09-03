# ML Intrusion Detection Dashboard

An undergraduate Telecommunications Engineering project that investigates whether a reduced set of network-flow features can preserve useful intrusion-detection performance while lowering the resource cost of inference on modest hardware.

The project classifies three flow labels from the CIC-DDoS2019 dataset:

- `BENIGN`
- `SYN`
- `UDP`

It uses one fixed Random Forest configuration in two controlled conditions:

1. All usable numeric flow features.
2. The 15 highest-ranked features selected from training data using mutual information.

The final model is demonstrated through a Streamlit dashboard that performs controlled replay of prepared CSV flow records. The dashboard is not a live network monitor and this project does not generate denial-of-service traffic.

## Project status

The governed data pipeline, training-only feature selection, frozen model
comparison, one-time external evaluation, replay-sample generator, and
controlled-replay dashboard are implemented. The full model met the quality
targets; the frozen 15-feature model did not. The dashboard preserves that
negative result in a permanent warning and is limited to prepared CSV replay.
Development validation is not presented as final performance evidence.

## Aim

To design and implement a machine-learning intrusion-detection dashboard that identifies benign, SYN-flood, and UDP-flood network flows using reduced flow features and controlled traffic replay.

## Research questions

1. Which flow features are most informative for distinguishing `BENIGN`, `SYN`, and `UDP` traffic?
2. How accurately does the Random Forest classify the three classes using 15 selected features?
3. How does reduced-feature performance differ from full-feature performance?
4. Does feature reduction improve efficiency on the target computer?
5. Can the reduced model support a controlled-replay dashboard?

## Experimental design

The input feature set is the independent condition. The algorithm, hyperparameters, training records, test records, and random seed remain fixed so that the comparison isolates the effect of feature reduction.

| Element | Full condition | Reduced condition |
| --- | --- | --- |
| Algorithm | Random Forest | Random Forest |
| Hyperparameters | Identical | Identical |
| Training records | Same records | Same records |
| Final test records | Same records | Same records |
| Input features | All usable numeric features | Top 15 training-derived features |

The primary quality measure is macro F1-score. Per-class recall, confusion matrices, false-positive rates, prediction latency, throughput, memory change, and stored model size are also measured.

## Frozen external-test result

The single hash-gated external evaluation used 60,000 balanced records from the
independent 2019-03-11 collection day (20,000 per class).

| Condition | Features | Accuracy | Macro F1 | BENIGN recall | SYN recall | UDP recall | Median time | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 64 | 0.9761 | 0.9761 | 0.9991 | 0.9295 | 0.9997 | 0.1048 s | 572,690 flows/s |
| Reduced | 15 | 0.6650 | 0.5547 | 0.9956 | 0.0000 | 0.9994 | 0.0706 s | 849,547 flows/s |

The reduced condition improved median latency and throughput, but its macro-F1
loss was 0.4214 and its SYN-recall loss was 0.9295. It therefore failed the
predeclared quality margins and must not be represented as an adequate SYN/UDP
detector. This negative result is retained without post-test feature reselection
or tuning. Complete metrics and all timing repetitions are under
`results/final_external_test/`.

## Data and leakage controls

The project uses the flow CSV files from the [CIC-DDoS2019 dataset](https://www.unb.ca/cic/datasets/ddos-2019.html). Raw data is not committed to this repository.

Key controls include:

- Keep the dataset's training and testing days physically separate.
- Fit imputation, feature selection, and models using training-day data only.
- Use the testing day once for the frozen final comparison.
- Remove identifiers and shortcut fields before modelling, including flow IDs, IP addresses, timestamps, ports, and protocol.
- Use `BENIGN`, `SYN`, and `UDP` as the effective targets. The verified training
  label `DRDOS_UDP` is mapped explicitly to `UDP`; raw provenance remains in the
  manifest. Do not merge any other similarly named attack, including UDP-Lag.
- Use `random_state=42` for reproducible random operations.
- Record source-file and model-artifact SHA-256 hashes.

Users must obtain the dataset from its official source and follow the current terms of use.

After placing confirmed collection-day CSVs in the separate raw-data folders,
generate the governed source manifest before preparation:

```powershell
python src/inspect_dataset.py `
  --download-date YYYY-MM-DD `
  --train-official-day "CONFIRMED TRAINING DAY" `
  --test-official-day "CONFIRMED TESTING DAY"
```

See `docs/DATASET.md` for the acquisition, placement, hashing, and completion
rules. The script scans only label columns and does not expose the external-test
feature distributions.

Prepare the bounded, balanced modelling datasets after the manifest gate passes:

```powershell
python src/prepare_data.py
```

This scans CSVs in 50,000-row chunks, retains a seeded uniform sample with
maximums of 50,000 rows per training class and 20,000 per external-test class,
balances each role down to its smallest available target class without row
duplication, removes
identifier and shortcut fields, keeps shared training-varying numeric features,
and writes ignored Parquet data plus committed JSON audit metadata.

## Repository structure

```text
ML-Intrusion-Detection-Dashboard/
|-- artifacts/                 # Feature metadata and trusted model bundles
|-- dashboard/
|   `-- app.py                 # Streamlit controlled-replay dashboard
|-- data/
|   |-- interim/               # Intermediate data (not committed)
|   |-- processed/             # Prepared Parquet files (not committed)
|   |-- raw/
|   |   |-- train_day/         # Official training-day CSVs (not committed)
|   |   `-- test_day/          # Official testing-day CSVs (not committed)
|   `-- replay/                # Prepared demonstration inputs (not committed)
|-- docs/                      # Project documentation
|-- results/                   # Dated metrics, plots, and benchmark records
|-- src/
|   |-- evaluate_models.py
|   |-- explore_data.py
|   |-- make_replay_sample.py
|   |-- prepare_data.py
|   |-- select_features.py
|   `-- train_models.py
|-- tests/
|   `-- test_pipeline.py
|-- environment-lock.txt
|-- README.md
`-- requirements.txt
```

Raw, processed, and replay data remain local and are excluded from Git. Model
files also remain local; their names and SHA-256 hashes are recorded in the
committed artifact manifests.

## Target environment

The implementation is designed for a computer with approximately 8 GB RAM and a dual-core CPU. Data preparation will use chunked CSV reads, per-class caps, compressed Parquet storage, and a maximum of two model-training jobs.

### Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The direct dependency contract is recorded in `requirements.txt`. After installation and verification, exact resolved versions are recorded in `environment-lock.txt`.

## Reproducible workflow

From a prepared environment, run:

```powershell
python src/prepare_data.py
python src/explore_data.py
python src/select_features.py
python src/train_models.py
pytest -q
python src/evaluate_models.py
python src/make_replay_sample.py
streamlit run dashboard/app.py
```

Every important run should write to a new dated results directory. Results used in the report must not be overwritten.

## Dashboard scope

The Streamlit interface:

- Load a fixed, trusted reduced-feature model artifact.
- Accept prepared flow-record CSV files, not model uploads.
- Validate required columns and enforce a safe row limit.
- Replay records in configurable batches.
- Display processed-flow counts, predicted classes, alerts, latency, and throughput.
- Show a recent-alert table and allow the prediction log to be downloaded.
- Clearly label the demonstration as controlled replay.

## Safety and ethical use

This is a defensive academic project. It analyzes an established research dataset and does not:

- Generate or transmit denial-of-service traffic.
- Probe external or operational systems.
- Process packet captures as part of the implementation.
- Claim that stored CSV replay is production live monitoring.
- Accept untrusted serialized model uploads.

Python `joblib` artifacts can execute code while loading. Only locally produced, verified model artifacts should be used.

## Reproducibility and evidence

The project retains:

- Dataset manifests and source-file hashes.
- Package versions and fixed configuration values.
- Data-preparation summaries and usable feature names.
- The complete mutual-information ranking and selected 15 features.
- Validation and external-test metrics.
- All timing repetitions and resource measurements.
- Model hashes and sizes.
- Automated and functional test records.
- Dashboard screenshots and downloaded replay logs.
- Known limitations and any failed acceptance criteria.

## Author

- Bernard Boakye Appiah
- Student, Humber College, Canada
- N10036999@humber.ca

## Limitations

- Evaluation on CIC-DDoS2019 does not prove generalization to another network or time period.
- Feature importance and mutual information do not establish causation.
- A benchmark on stored flow records does not represent end-to-end live detection latency.
- Random Forest is the selected project algorithm; the study does not claim that it is superior to every alternative.

## Contributing

Keep changes reproducible, focused, and reviewable. Do not commit raw datasets, secrets, virtual environments, caches, or unapproved large model artifacts. Run the relevant tests before committing and document any change to a frozen experimental decision.

