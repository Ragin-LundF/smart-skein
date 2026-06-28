# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-06-28

- Fixed some performance issues in training for classification
- Added a new example, `logwatch`, for monitoring log files and extracting structured data with classification

## [1.0.0] - 2026-06-28

### First final release

- **Classification** — Naive Bayes and logistic regression classifiers for assigning labels to text records. Supports active learning so the model improves over time as more labeled data is provided.
- **Privacy-preserving features** — personal data never leaves the system in plain text. Feature extraction uses one-way hashing, so the original content cannot be reconstructed from a saved model.
- **Text foundation** — shared text normalization, broken-word repair (e.g. `"apart ment" → "apartment"`), and a typed tokenizer that turns free text into pattern signatures.
- **Extraction** — pull structured values from text using typed-token patterns and slot filling.
- **PostgreSQL storage** — optional persistence adapter for storing labeled observations in a database, with AES-256-GCM encryption at rest.
- **CLI tools** — command-line interface for interactive data labeling (`label`), batch classification (`predict`), and model export (`export`). Supports CSV input and output with configurable delimiters.
- **Duplicate filtering** — repeated ingestion of the same record is detected and silently ignored, keeping the model and saved file size stable.
- **Model file format** — trained models are saved to a single `.skein` file and can be restored without retraining from source data.
- **BOM** — a Bill of Materials module is provided for consumers who want to align all Skein module versions in one place.
