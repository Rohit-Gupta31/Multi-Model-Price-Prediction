# Amazon_ML_Challenge
## MJAVE Price Regression (PyTorch)

This repository contains a PyTorch reimplementation of a MJAVE-style multimodal model for price regression (images + text). The codebase provides utilities to precompute embeddings (text and image), a "fast" training path that uses those embeddings, scripts to replace only text embeddings with your own DeBERTa weights, and multiple inference modes.

Summary
- Model: multimodal (text + image) MJAVE-style network implemented in `custom_multimodel/mjave_price_model.py`.
- Task: regression (predicting price). The model outputs log-price values during training/prediction.
- Best reported validation SMAPE achieved in this workspace: 45.2% (user-reported).

Quick structure
- `custom_multimodel/` - model implementation and helpers (e.g. `mjave_price_model.py`).
- `precompute_embeddings.py` - produce training embeddings (text + image) from raw data.
- `precompute_test_embeddings.py` - (rewritten) produce test embeddings using the same extractors.
- `replace_text_embeddings.py` - replace only the text embeddings in an existing embedding folder using a DeBERTa checkpoint (useful if you already have image embeddings).
- `train_mjave_price_fast.py` - fast training loop that loads precomputed embeddings from `./embeddings`.
- `inference_mjave_exact.py`, `test_mjave_inference.py`, `test_mjave_inference_live.py` - inference scripts (precomputed vs on-the-fly).
- `calculate_smape.py` - evaluation helper (note: handles log-space correctly).

Important invariants
- Text tokenization is run with MAX_LENGTH=256 which produces 254 content token vectors after removing special tokens (the code expects text feature tensors shaped like `[N, 254, 768]`).
- Image regional features: `[N, 49, 2048]`. Image global features: `[N, 2048]`.
- Model outputs and metrics expect log-space prices internally. `compute_smape()` expects log-space arrays (it runs expm1 internally). When evaluating against real prices saved to CSV, use `np.log1p(price)` before calling the training SMAPE routine.

