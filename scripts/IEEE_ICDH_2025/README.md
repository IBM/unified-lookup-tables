## Aligning foundation models on encoded synthetic omic data for patient stratification (IEEE ICDH 2025)

This folder contains code and config files associated with the IEEE ICDH 2025 paper titled [Aligning foundation models on encoded synthetic omic data for patient stratification](). 

### Data augmentation

The paper follows the data augmentation strategy described in the paper ["Phenotype driven data augmentation methods for transcriptomic data"](https://academic.oup.com/bioinformaticsadvances/article/5/1/vbaf124/8142420) to generate enough samples for LLM training. To reproduce the augmentation, please follow the steps described below:
1. Go to the [GitHub repo](https://github.com/PaccMann/transcriptomic_signature_sampling) of the original method. 
2. Switch to the branch ``ieee_icdh``.
3. Under ``scripts``, run the file ``augment_modgp_rnaseq.py`` with the appropriate file paths.

The data used for experiments in the paper can be found on [Zenodo](https://doi.org/10.5281/zenodo.15641421) (https://doi.org/10.5281/zenodo.15641421) in ``data.zip``.   

### Generating ULTs for Transcriptomic data

To create a lookup table for transcriptomic data, the paper uses the ``UnicodeSeriesCompansionTransform`` class to first compress the patchified data, followed by sorted frequency mapping to unicode characters. 

The resulting lookup table from the paper can be found in ``lut-rna-42-series-p10cm1mu256.json``.

### LLM Training

The paper uses a pre-trained SmolLM-1.7B for fine-tuning on encoded transcriptomic data for colorectal cancer subtyping. The Trainer and evaluation arguments can be found in the ``config.yaml`` file. 

To run a training, simply pass "multiomics" as the ``dataset_name`` and pass *scripts/IEEE_ICDH_2025/config.yaml* as the ``train_args_path`` to the ``training_ult.py`` file as follows:

```console
poetry run scripts/python training_ult.py --dataset_name multiomics --train_args_path scripts/IEEE_ICDH_2025/config.yaml
```

### Evaluation

Functions to compute correlation between distances in the input space and encoded strong space can be found in ``feature_distance_correlation.py``. 

The results shown in the paper for each experimental repetition can be found on [Zenodo](https://doi.org/10.5281/zenodo.15641421) (https://doi.org/10.5281/zenodo.15641421) in ``results_mu256.zip``.   

