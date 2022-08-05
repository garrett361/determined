#!/bin/bash
read -p "Master URL: " MASTER_URL
read -p "Number of base models to use (Default 2 3 4 5): " NUM_BASE_MODELS
read -p "Number of Ensembles per strategy (Default 100): " NUM_ENSEMBLES
read -p "Number of combinations for VBMC (Default 512): " NUM_COMBINATIONS
read -p "Number of training epochs, when applicable (Default 1): " EPOCHS
read -p "Learning rate for SGD training (Default .001): " LR
read -p "Model criteria, small, top or all (Default small): " MODEL_CRITERIA
read -p "Dataset name (Default imagenette2-160): " DATASET_NAME

echo -e "\nThis will run all ensembling experiments with the following parameters:\n"

echo "Master URL: $MASTER_URL"
echo "Number of base models: ${NUM_BASE_MODELS:=2 3 4 5}"
echo "Number of Ensembles per strategy: ${NUM_ENSEMBLES:=100}"
echo "Number of combinations for VBMC: ${NUM_COMBINATIONS:=512}"
echo "Number of training epochs for SGD training: ${EPOCHS:=1}"
echo "Learning rate for SGD training: ${LR:=.001}"
echo "Model criteria: ${MODEL_CRITERIA:=small}"
echo "Dataset name: ${DATASET_NAME:=imagenette2-160}"
echo ""

read -p "Do you want to proceed? (yes/NO) " PROCEED_CONFIRM

case $PROCEED_CONFIRM in
	yes ) echo Submitting;;
	* ) echo Exiting;
		exit 1;;
esac

# Run $NUM_ENSEMBLES ensembles for every X-model ensembles, X in "$NUM_BASE_MODELS"
python3 script_timm.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_ENSEMBLES" -es naive naive_logits most_confident majority_vote -mc "$MODEL_CRITERIA" --no_safety_check --delete_unvalidated
python3 script_timm.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_ENSEMBLES" -es naive_temp naive_logits_temp most_confident_temp super_learner_probs super_learner_probs_temp super_learner_logits -mc "$MODEL_CRITERIA" --no_safety_check -e "$EPOCHS"
python3 script_timm.py -m "$MASTER_URL" -nbm "$NUM_BASE_MODELS" -ne "$NUM_ENSEMBLES" -es vbmc vbmc_temp -mc "$MODEL_CRITERIA" -nc "$NUM_COMBINATIONS" --no_safety_check