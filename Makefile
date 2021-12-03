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
CONFIG_PATH=./experiments/configs/
CONFIG_NAME=seld.yml
OUTPUT=./outputs   # Directory to save output
EXP_SUFFIX=_test   # the experiment name = CONFIG_NAME + EXP_SUFFIX
RESUME=False
GPU_NUM=0  # Set to -1 if there is no GPU

.phony: train
train:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/train.py --exp_config="${CONFIG_PATH}${CONFIG_NAME}" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX) --resume=$(RESUME)

.phony: inference
inference:
	PYTHONPATH=$(shell pwd) CUDA_VISIBLE_DEVICES="${GPU_NUM}" python experiments/inference.py --exp_config="${CONFIG_PATH}${CONFIG_NAME}" --exp_group_dir=$(OUTPUT) --exp_suffix=$(EXP_SUFFIX)


# Evaluate
OUTPUT_DIR=./outputs/crossval/foa/salsa/seld_test/outputs/submissions/original/foa_test
GT_ROOT_DIR=/data/seld_dcase2021/task3
IS_EVAL_SPLIT=False

.phony: evaluate
evaluate:
	PYTHONPATH=$(shell pwd) python experiments/evaluate.py  --output_dir=$(OUTPUT_DIR) --gt_meta_root_dir=$(GT_ROOT_DIR) --is_eval_split=$(IS_EVAL_SPLIT)
