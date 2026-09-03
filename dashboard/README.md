# Dashboard

`app.py` is the Streamlit controlled-replay interface. It accepts only prepared
flow-record CSV files, validates required feature columns, imposes a 10,000-row
limit, and loads only the hash-verified locally generated reduced model.

The frozen 15-feature model failed its external-test quality gate, including
zero SYN recall. The interface therefore displays a permanent prominent warning
and presents the model only as a coursework demonstration—not an operational
detector or live monitor.

Run from the repository root:

```powershell
streamlit run dashboard/app.py
```

Repository configuration disables Streamlit usage-statistics collection so the
local demonstration does not require writes to a user-profile telemetry file.

To create the immutable replay evidence bundle after generating the sample:

```powershell
python src/run_replay_evidence.py
```
