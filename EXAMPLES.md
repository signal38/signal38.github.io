// Local only. Don't commit

⏺ Here are 11 concrete examples worth exploring:

  ---
  Knowledge graphs / RAG
  - Talking to GDELT Through Knowledge Graphs (2025) — converts GKG2 into a KG and benchmarks RAG strategies for reasoning over events the LLM hasn't seen,
  using the Francis Scott Key Bridge collapse as a live test case.

  Finance
  - EU Sovereign Bond Markets — LSTM models trained on GCAM emotional indicators from Italian-language GDELT news to forecast BTP/Bund spreads. "Arousal" and
  "Hate" tone dimensions outperformed macro regressors.
  - Chinese Stock Market Sentiment (BBVA Research) — decomposed tone into 5 orthogonal dimensions; found lower coverage volume predicts higher next-day returns.

  Full-text reconstruction
  - Free Access to World News / gdeltnews (2025) — exploits GDELT's positional n-gram metadata to reconstruct full articles at 95% similarity. Sidesteps
  paywalls entirely.

  Media analysis
  - AI Watching Television News — object detection + face recognition on broadcast TV frames to quantify airtime divergence between CNN/MSNBC/Fox on the same
  story.
  - COVID Media Narrative — 4.9M minutes of TV + 2.5M minutes of radio analyzed via speech-to-text to track sub-narrative airtime shifts over 1,113 days.

  Conflict / protest
  - GDELT Conflict Dataset on Kaggle — 80M events filtered to conflict-only, used for ML forecasting benchmarked against ACLED and UCDP-GED.
  - Mapping Global Protest Trends — avoids naive event-counting by using protest share of total coverage as the signal.

  Methodology critique (worth reading)
  - Lifting the Veil on Big Data News Repositories — adversarial audit of GDELT's protest event pipeline vs. human-coded gold standard. Documents where the
  coding breaks and why.

  Tooling
  - gdeltPyR — parallelized Python fetcher with Spark-compatible output; handles the memory/scale problem (a single day is 400+ MB uncompressed).

  BigQuery longitudinal analysis
  - Towards Psychohistory — cross-tabbed correlation of 300+ relationship classes between country pairs over 35 years, all in BigQuery. The SQL templates are
  reusable.

  The n-gram reconstruction paper and the bond market LSTM work are the two I'd call genuinely clever rather than just large-scale.
