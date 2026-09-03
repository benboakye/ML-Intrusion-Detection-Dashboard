"""Streamlit controlled-replay demonstration for the frozen reduced model."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from ml_nids.config import ARTIFACTS_DIR, DASHBOARD_MAX_ROWS
from ml_nids.preparation import DataPreparationError
from ml_nids.replay import (
    load_trusted_bundle,
    run_controlled_replay,
    validate_replay_frame,
)

st.set_page_config(page_title="SYN/UDP Flow Replay", page_icon="🛡️", layout="wide")
st.title("ML Intrusion Detection Dashboard")
st.caption("Controlled replay of prepared CIC-DDoS2019 flow records")
st.error(
    "Research limitation: the frozen 15-feature model failed the external-test "
    "quality gate and recorded 0.000 SYN recall. This interface is a coursework "
    "demonstration, not an operational detector or live monitor."
)


@st.cache_resource
def model_bundle():
    return load_trusted_bundle(
        ARTIFACTS_DIR, ARTIFACTS_DIR / "model_manifest.json", "reduced"
    )


try:
    bundle = model_bundle()
except (DataPreparationError, OSError, ValueError) as exc:
    st.error(f"Trusted model could not be loaded: {exc}")
    st.stop()

with st.sidebar:
    st.header("Frozen model")
    st.write("Condition: Reduced experimental model")
    st.write(f"Selected features: {len(bundle['features'])}")
    st.code(bundle["training_sha256"][:16] + "…", language=None)
    st.caption(f"Upload limit: {DASHBOARD_MAX_ROWS:,} prepared rows")

upload = st.file_uploader("Prepared replay CSV", type=["csv"])
batch_size = st.slider("Batch size", 100, 5_000, 1_000, 100)

if upload is not None:
    try:
        uploaded = pd.read_csv(upload, low_memory=False)
        inputs = validate_replay_frame(uploaded, bundle["features"])
    except (DataPreparationError, ValueError, pd.errors.ParserError) as exc:
        st.error(str(exc))
        st.stop()
    st.success(f"Validated {len(inputs):,} prepared flow rows.")

    if st.button("Run controlled replay", type="primary"):
        result, performance = run_controlled_replay(bundle, inputs, batch_size)
        st.session_state["replay_result"] = result
        st.session_state["replay_performance"] = performance

if "replay_result" in st.session_state:
    result = st.session_state["replay_result"]
    performance = st.session_state["replay_performance"]
    counts = result["prediction"].value_counts().reindex(
        ["BENIGN", "SYN", "UDP"], fill_value=0
    )
    first, second, third, fourth, fifth = st.columns(5)
    first.metric("Processed", f"{len(result):,}")
    second.metric("Benign", f"{int(counts['BENIGN']):,}")
    third.metric("SYN alerts", f"{int(counts['SYN']):,}")
    fourth.metric("UDP alerts", f"{int(counts['UDP']):,}")
    fifth.metric("Throughput", f"{performance['flows_per_second']:,.0f}/s")

    st.subheader("Predicted class counts")
    st.bar_chart(counts)
    st.subheader("Recent alerts")
    st.dataframe(result.loc[result["alert"]].tail(20), use_container_width=True)
    st.download_button(
        "Download prediction log",
        result.to_csv(index=False),
        file_name="replay_predictions.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "No packets or attacks are generated. Uploaded data remains flow-record CSV "
    "content processed within this local Streamlit session."
)
