## Aligning foundation models on encoded synthetic omic data for patient stratification (IEEE ICDH 2025)

This folder contains code and config files associated with the IEEE ICDH 2025 paper titled [Aligning foundation models on encoded synthetic omic data for patient stratification](). 

### Data augmentation

The paper follows the data augmentation strategy described in the paper ["Phenotype driven data augmentation methods for transcriptomic data"](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf124/8142420) to generate enough samples for LLM training. To reproduce the augmentation, please follow the steps described below:
1. Go to the [GitHub repo](https://github.com/PaccMann/transcriptomic_signature_sampling) of the original method. 
2. Switch to the branch ``ieee_icdh``.
3. Under ``scripts``, run the file ``augment_modgp_rnaseq.py`` with the appropriate file paths.

### Generating ULTs for Transcriptomic data

To create a lookup table for transcriptomic data, the paper uses the ``UnicodeSeriesCompansionTransform`` class to first compress the patchified data, followed by sorted frequency mapping to unicode characters. 

The resulting lookup table from the paper can be found in ``lut-rna-42-series-p10cm1mu256.json``.

### LLM Training

Pre-trained SmolLM-1.7B was used for fine-tuning on encoded transcriptomic data for colorectal cancer subtyping. The Trainer and evaluation arguments can be found in the ``config.yaml`` file. 

### Evaluation

Functions to compute correlation between distances in the input space and encoded strong space can be found in ``feature_distance_correlation.py``. 

``MultiOmicsEvaluator`` class processes the predicted string to retrieve the target for metric computation.
