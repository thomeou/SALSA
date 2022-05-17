# Feature extraction: to extract feature linspeciv, melspeciv, linspecgcc, melspecgcc
FEATURE_CONFIG=./dataset/configs/tnsse2021_feature_config.yml
FEATURE_TYPE=linspeciv

.phony: feature
feature:
	PYTHONPATH=$(shell pwd) python dataset/feature_extraction.py --data_config=$(FEATURE_CONFIG) \
	--feature_type=$(FEATURE_TYPE)


# SALSA feature extraction
SALSA_CONFIG=./dataset/configs/tnsse2021_salsa_feature_config.yml

.phony: salsa
salsa:
	PYTHONPATH=$(shell pwd) python dataset/salsa_feature_extraction.py --data_config=$(SALSA_CONFIG)


# SALSA-LITE feature extraction
SALSA_LITE_CONFIG=./dataset/configs/tnsse2021_salsa_lite_feature_config.yml
SALSA_LITE_FEATURE_TYPE=salsa_lite  # salsa_lite | salsa_ipd

.phony: salsa-lite
salsa-lite:
	PYTHONPATH=$(shell pwd) python dataset/salsa_lite_feature_extraction.py --data_config=$(SALSA_LITE_CONFIG) --feature_type=$(SALSA_LITE_FEATURE_TYPE)


# Training and inference
CONFIG_DIR=./experiments/configs
OUTPUT=./outputs   # Directory to save output
EXP_SUFFIX=_test   # the experiment name = CONFIG_NAME + EXP_SUFFIX
RESUME=False
GPU_NUM=0  # Set to -1 if there is no GPU

.phony: train-salsa
train-salsa:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/train.py --exp_config="${CONFIG_DIR}/seld.yml" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX) --resume=$(RESUME)

.phony: inference-salsa
inference-salsa:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/inference.py --exp_config="${CONFIG_DIR}/seld.yml" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX)

.phony: train-salsa-lite
train-salsa-lite:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/train.py --exp_config="${CONFIG_DIR}/seld_salsa_lite.yml" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX) --resume=$(RESUME)

.phony: inference-salsa-lite
inference-salsa-lite:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/inference.py --exp_config="${CONFIG_DIR}/seld_salsa_lite.yml" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX)

# Evaluate
OUTPUT_DIR=./outputs/crossval/mic/salsa/seld_test/outputs/submissions/original/mic_test
GT_ROOT_DIR=./dataset/data
IS_EVAL_SPLIT=False

.phony: evaluate
evaluate:
	PYTHONPATH=$(shell pwd) python experiments/evaluate.py  --output_dir=$(OUTPUT_DIR) --gt_meta_root_dir=$(GT_ROOT_DIR) --is_eval_split=$(IS_EVAL_SPLIT)