# CIC-DDoS2019 acquisition and governance

Use only the flow CSV files obtained from the official University of New
Brunswick CIC-DDoS2019 page:

https://www.unb.ca/cic/datasets/ddos-2019.html

The official page permits redistribution but requires citation to the dataset
and its related paper. This project does not redistribute the raw files. Keep
all archives, CSVs, PCAPs, and generated Parquet files outside Git.

## 1. Record the acquisition

Record the download date and the official collection-day name associated with
each archive. Do not infer training and external-test roles from filenames. The
roles must be confirmed before files are copied into the project.

## 2. Place the flow CSVs

```text
data/raw/train_day/   # Confirmed development/training collection day
data/raw/test_day/    # Confirmed independent external-test collection day
```

Nested directories are allowed. The inspector searches recursively.

## 3. Generate the manifest

Replace the collection-day arguments with the exact terminology recorded at
download time:

```powershell
python src/inspect_dataset.py `
  --download-date YYYY-MM-DD `
  --train-official-day "CONFIRMED TRAINING DAY" `
  --test-official-day "CONFIRMED TESTING DAY"
```

The command scans only the label column, counts rows, records normalized label
values, calculates SHA-256 for every CSV, and writes
`docs/dataset_manifest.csv`. It returns a non-zero status if either partition
does not contain all three required labels: `BENIGN`, `SYN`, and `UDP`.

## Completion gate

- The official source and download date are recorded.
- Training and external-test folders are physically separate.
- Every CSV has a SHA-256 hash, row count, role, collection day, and label list.
- Exact `BENIGN`, `SYN`, and `UDP` labels are observed in both roles.
- No raw data, archive, PCAP, or serialized model is staged in Git.
