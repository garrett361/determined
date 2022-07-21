#!/bin/bash
read -p "Master URL: " MASTER_URL
read -p "Number of base models to use, beyond the default 1 and 2 (Default 3 4 5): " NUM_BASE_MODELS
read -p "Number of Ensembles per strategy, when using > 2 base models (Default 100): " NUM_ENSEMBLES
read -p "Number of combinations for VBMC (Default 512): " NUM_COMBINATIONS
read -p "Number of training epochs for SGD training (Default 3): " EPOCHS
read -p "Learning rate for SGD training (Default .001): " LR
read -p "Model criteria, small, top or all (Default small): " MODEL_CRITERIA
read -p "Dataset name (Default imagenette2-160): " DATASET_NAME

echo -e "\nThis will run all ensembling experiments with the following parameters:\n"

echo "Master URL: $MASTER_URL"
echo "Number of base models: ${NUM_BASE_MODELS:=3 4 5}"
echo "Number of Ensembles per strategy, when using > 2 base models: ${NUM_ENSEMBLES:=100}"
echo "Number of combinations for VBMC: ${NUM_COMBINATIONS:=512}"
echo "Number of training epochs for SGD training: ${EPOCHS:=3}"
echo "Learning rate for SGD training: ${LR:=.001}"
echo "Model criteria: ${MODEL_CRITERIA:=small}"
echo "Dataset name: ${DATASET_NAME:=imagenette2-160}"
echo ""

read -p "Do you want to proceed? (yes/NO) " yn

case $yn in
	yes ) echo Submitting;;
	* ) echo Exiting;
		exit 1;;
esac

# Single model baselines
python3 script.py -m "$MASTER_URL" -nbm 1 -ne -1 -es naive -mc "$MODEL_CRITERIA" --no_safety_check

# Run all 2-model ensembles, by default
python3 script.py -m "$MASTER_URL" -nbm 2 -ne -1 -es naive naive_temp naive_logits naive_logits_temp most_confident most_confident_temp majority_vote -mc "$MODEL_CRITERIA" --no_safety_check
python3 script.py -m "$MASTER_URL" -nbm 2 -ne -1 -es vbmc vbmc_temp -mc "$MODEL_CRITERIA" -nc "$NUM_COMBINATIONS" --no_safety_check
python3 script.py -m "$MASTER_URL" -nbm 2 -ne -1 -es super_learner_probs super_learner_logits -mc "$MODEL_CRITERIA" -e "$EPOCHS" -lr "$LR" --no_safety_check

# Run $NUM_ENSEMBLES ensembles for every X-model ensembles, X in "$NUM_BASE_MODELS"
python3 script.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_COMBINATIONS" -es naive naive_temp naive_logits naive_logits_temp most_confident most_confident_temp majority_vote -mc "$MODEL_CRITERIA" --no_safety_check
python3 script.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_COMBINATIONS" -es vbmc vbmc_temp -mc "$MODEL_CRITERIA" -nc "$NUM_COMBINATIONS" --no_safety_check
python3 script.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_COMBINATIONS" -es super_learner_probs super_learner_logits -mc "$MODEL_CRITERIA" -e "$EPOCHS" -lr "$LR" --no_safety_check