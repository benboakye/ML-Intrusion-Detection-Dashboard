# Frozen external-test evidence

These files record the project's single final evaluation on the independently
prepared 2019-03-11 test-day data. The run was gated by the hashes in the data
preparation summary, model manifest, and pre-test freeze record.

The 64-feature Random Forest achieved macro F1 0.9761. The frozen 15-feature
condition achieved macro F1 0.5547 and zero SYN recall. Although the reduced
condition improved median prediction time and throughput, it failed the frozen
macro-F1 and SYN-recall acceptance margins.

No feature reselection or model retuning may be justified from these results.
The JSON file is the complete machine-readable record; the CSV files provide a
compact comparison and all five timing repetitions, and the PNG files visualize
the confusion matrices.
