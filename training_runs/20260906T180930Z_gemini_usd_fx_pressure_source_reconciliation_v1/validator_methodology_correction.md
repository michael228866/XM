# Validator Methodology Correction

The initial validator correctly found that no source passed the strict data-certification gate, but it also reported methodology failures for causal reconstruction and feature formulas.

Those methodology failures were caused by validator control flow: when `certified_provider` was `null`, the validator skipped reconstruction of the last attempted single-source Dukascopy candidate and compared the saved matrix with an all-NaN placeholder.

The validator was corrected to reconstruct that preserved candidate for methodology checks while keeping certification separate. No source data, timestamp blocks, feature matrix, gap classification, provider certification, or `data_ready_for_training` value was changed. The initial validator script and output are retained for provenance.
