# Validator methodology corrections

The first validator attempt failed on two validator-only assumptions:

- it read sklearn constructor attributes after loading a raw XGBoost model, although those
  attributes are not restored by that serialization format;
- it required near-bitwise Spearman agreement after net-R was intentionally serialized as
  float32 in the retained OOF artifact.

The second attempt corrected the float32 tolerance and independently reconciled fold and pooled
ranking, but still tried to infer training-only tree parameters from the raw model configuration.

The final validator instead checks all six distinct model hashes, B0/B1 feature schemas, fitted
tree count, objective, immutable training-script identity, immutable dependency hashes, paired
fold provenance, seed declarations, and full OOF/S5 reconstruction. The float32 ranking tolerance
is fixed at 1e-5 and applies uniformly to both models and all folds before validation.

No model was retrained. No OOF probability, trade ledger, threshold, macro feature, candidate
decision, or strategy metric was changed between validator attempts.
