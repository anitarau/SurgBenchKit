#!/bin/bash
# Define model abbreviations
declare -A model_abbreviations=(
  ["PaliGemma"]="PG"
  ["Qwen2-VL"]="QW"
  ["Gemini"]="GM"
  ["GeminiPro1-5"]="GM"
  ["GPT4o"]="GP"
  ["llava_next_vicuna_7b"]="LV"
  ["CLIP"]="CL"
  ["OpenCLIP"]="OCL"
  ["SurgVLP"]="SV"
)

# Define task abbreviations
declare -A task_abbreviations=(
  ["avos_action_prediction"]="avo_ac"
  ["avos_action_prediction_fewshot"]="avo_ac_fs"
  ["cholec80_phase_detection"]="ch8_ph"
  ["cholec80_phase_detection_fewshot"]="ch8_ph_fs"
  ["cholec80_tool_presence"]="ch8_tp"
  ["cholec80_tool_presence_fewshot"]="ch8_tp_fs"
  ["cholect45_triplet"]="cho45_tr"
  ["cholect45_triplet_fewshot"]="cho45_tr_fs"
  ["dresden_anatomy_presence"]="dre_an"
  ["dresden_anatomy_presence_fewshot"]="dre_an_fs"
  ["endoscapes_cvs_assessment"]="end_cvs"
  ["endoscapes_cvs_assessment_fewshot"]="end_cvs_fs"
  ["endoscapes_object_detection"]="end_obj"
  ["endoscapes_object_detection_fewshot"]="end_obj_fs"
  ["endoscapes_tool_detection"]="end_td"#
  ["heichole_action_prediction"]="hei_ac"
  ["heichole_action_prediction_fewshot"]="hei_ac_fs"
  ["heichole_phase_detection"]="hei_ph"
  ["heichole_phase_detection_fewshot"]="hei_ph_fs"
  ["heichole_tool_presence"]="hei_tp"
  ["heichole_tool_presence_fewshot"]="hei_tp_fs"
  ["multibypass140_phase_detection"]="mb140_ph"
  ["avos_object_detection"]="avo_obj"
  ["intermountain_cvs_assessment"]="int_cvs"
)
# Define the parameter sets
infer="1"  # set 1 to get predictions, and 0 to compute metrics of existing predictions only
models=("GeminiPro1-5" "GPT4o" "CLIP" "OpenCLP" "Qwen2-VL" "PaliGemma" "llava_next_vicuna_7b")

tasks=("cholec80_tool_presence" "cholect45_triplet" "endoscapes_cvs_assessment" "heichole_action_prediction" "heichole_tool_presence" "avos_action_prediction" "cholec80_phase_detection" "heichole_phase_detection" "multibypass140_phase_detection")


# Base paths
output_dir="/path/to/logs"
script_dir="/path/to/SurgBenchKit/scripts"

# Create script directory if it doesn't exist
mkdir -p "$script_dir"

# Loop over all combinations of model and task
for model in "${models[@]}"; do
  for task in "${tasks[@]}"; do
    # Get abbreviations
    model_abbr="${model_abbreviations[$model]}"
    task_abbr="${task_abbreviations[$task]}"

    # Generate unique eval.sh script for each task 
    eval_script_path="$script_dir/eval_${model_abbr}_${task_abbr}.sh"
    cat <<EOF > "$eval_script_path"
#!/usr/bin/env bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate vlm
cd /path/to/SurgBenchKit
export PYTHONPATH=Train:\${PYTHONPATH}
export HF_HOME="/path/to/where/to/save/model/weights/"
export GOOGLE_API_KEY="XXX"
python /path/to/SurgBenchKit/eval.py model=$model task=$task model.infer=infer_data_api exp_name=all_test
EOF
    chmod +x "$eval_script_path"

    # Determine GPU type based on the model
    if [[ "$model" == "Qwen2-VL" ]]; then
      gpu_type="a6000:1"
    elif [[ "$model" == "SurgVLP" ]]; then
      gpu_type="a6000:1"
    elif [[ "$model" == "PaliGemma" ]]; then
      gpu_type="a6000:1"
    elif [[ "$model" == "GeminiPro1-5" ]]; then
      gpu_type="0"
    elif [[ "$model" == "GPT4o" ]]; then
      gpu_type="0"
    elif [[ "$model" == "llava_next_vicuna_7b" ]]; then
      gpu_type="a6000:1"
    else
      gpu_type="1"
    fi

    if [[ "$infer" == "0" ]]; then
      gpu_type="0"
    fi

    # Generate sbatch script
    sbatch_script_path="$script_dir/${model_abbr}_${task_abbr}.sh"
    cat <<EOF > "$sbatch_script_path"
#!/bin/bash
#SBATCH -p pasteur
#SBATCH --mem=8gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:$gpu_type
#SBATCH --account=pasteur
#SBATCH --output=$output_dir/${model}_${task}_out.txt
#SBATCH --error=$output_dir/${model}_${task}_err.txt
bash $eval_script_path
EOF

    # Submit the job
    sbatch "$sbatch_script_path"

    # Optionally, add a short delay between submissions
    sleep 1
  done
done
