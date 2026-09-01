# Data workspace

Dataset files are local inputs and are not committed to Git.

Expected layout after obtaining the official CIC-DDoS2019 flow CSVs:

```text
data/
|-- raw/
|   |-- train_day/     # Official training-day CSV files
|   `-- test_day/      # Official testing-day CSV files
|-- interim/           # Temporary preparation outputs
|-- processed/         # train.parquet and external_test.parquet
`-- replay/            # Generated replay input and separate truth files
```

Do not guess file roles from archive names. The acquisition stage will inspect
the `Label` values, record the official collection day, and generate a manifest
and SHA-256 hashes before preparation begins.
