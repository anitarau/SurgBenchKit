

# A Toolkit for Evaluating Large Vision-Language Models for Surgery
This project builds upon VLMEvalKit © 2023 VLMEvalKit Authors, licensed under the Apache License 2.0.
Significant modifications and extensions were made by the SurgBenchKit authors for research purposes.
https://github.com/open-compass/VLMEvalKit
## 🏗️ Getting Started
### Step 1: Clone repo and install packages
```
git clone https://github.com/anitarau/SurgBenchKit.git
cd SurgBenchKit
pip install -e .
```

Also install this in your environment:
```
pip install flash-attn --no-build-isolation
pip install git+https://github.com/openai/CLIP.git
```

If you want to evaluate SurgVLP, we recommend using a seperate environment for it as Qwen and SurgVLP have a transformers version conflict. In the second environment follow there original instructions to set up a SurgVLP environment:
`pip install git+https://github.com/CAMMA-public/SurgVLP.git`

### Step 2: Add your API keys to .env file
For Gemini and GPT you'll need to set up an API key and add it to the `.env` file.
### Step 3: Prepare datasets
Download datasets from the original sources. Add their local paths in the task config files (e.g. `config/task/cholec80_phase_recognition.yaml`).
For Heichole we wrote a script to extract frames in `heichole_helpers.py` in case it helps.
## 📊  Inference and evaluation
We use hydra to manage task and dataset configurations. Each task and each model are configured in the `config` folder. You can change the default settings (model, task, etc.) in `config/config.yaml`.

To run a single task instance (one task on one data set with one model) use this:
```
python eval.py task=heichole_action_recognition model=GPT4o exp_name=experiment1
```
Here is a full list of all task instances currently supported:
```
python eval.py task=avos_action_recognition
python eval.py task=cholec80_phase_recogntion
python eval.py task=cholec80_tool_recogntion
python eval.py task=cholect45_triplet_recogntion
python eval.py task=desden_anatomy_presence
python eval.py task=endoscapes_cvs_assessment
python eval.py task=heichole_action_recognition
python eval.py task=heichole_phase_recognition
python eval.py task=heichole_tool_recognition
python eval.py task=multibypass140_phase_recognition

```

For few-shot experiments you can use this:
```
python eval.py task=heichole_action_recognition_fewshot model=GPT4o task.shots=five exp_name=experiment2_fiveshot
```

Running inference will automatically also run the evaluation. But if you have already ran the inference previously, and you only want to compute the metrics, you can use this: 
```
python eval.py task=heichole_action_recognition model=GPT4o exp_name=experiment1 cfg.eval_mode=eval_data
```

Coming soon: Video tasks including gesture recognition, skill assessment, and error recognition and detection.


This repo is currently configured for: 
- GeminiPro 1.5
- GPT-4o
- Qwen2-VL
- PaliGemma 1
- CLIP
- OpenCLIP
- SurgVLP
- InternVL 2.0
- Phi-3.5 Vision
- LLaVA-NeXT

# Experiment Sweep
If you want to run several tasks or several models at once, we prepared a script you can use to submit one sbatch job per task-instance to your local cluster: `run_all_combinations.sh`. Make sure you adjust all paths (search for "/path/to") and change the sbatch commands to be compatible with your cluster.

## Adding models or data
### Adding models
Our repo is built on VLMEvalKit which supports dozens more models. Please check the original repo for details. You can add models that are supported in VLMEvalKit by creating a model configuration in `config/model/`. Make sure to follow the naming conventions and other instructions in the original repo. 


## Adding datasets or tasks
To add a new dataset or task, you'll need to:
1. Add a dataloader
2. Create a task configuration in `config/task`
3. Import dataloader in `vlmeval/dataset/__init__.py`
4. Add dataloader to `data_map` in `vlmeval/config.py`
5. Add an evaluation loop specific to the dataset to `vlmeval/inference_surg.py`
6. Depending on whether the default option works for your dataset, add inference loop to `vlmeval/inference_surg.py`
7. Add a prompt to `prompts.py`


## 🖊️ Citation

If you find this work helpful, please consider **starring🌟** this repo.

If you would like to cite SurgBenchKit please use

```bib
@article{rau2025systematic,
  title={Systematic Evaluation of Large Vision-Language Models for Surgical Artificial Intelligence},
  author={Rau, Anita and Endo, Mark and Aklilu, Josiah and Heo, Jaewoo and Saab, Khaled and Paderno, Alberto and Jopling, Jeffrey and Holsinger, F Christopher and Yeung-Levy, Serena},
  journal={arXiv preprint arXiv:2504.02799},
  year={2025}
}
```
and also consider citing VLMEvalKit:
```bib
@inproceedings{duan2024vlmevalkit,
  title={Vlmevalkit: An open-source toolkit for evaluating large multi-modality models},
  author={Duan, Haodong and Yang, Junming and Qiao, Yuxuan and Fang, Xinyu and Chen, Lin and Liu, Yuan and Dong, Xiaoyi and Zang, Yuhang and Zhang, Pan and Wang, Jiaqi and others},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={11198--11201},
  year={2024}
}
```
<p align="right"><a href="#top">🔝Back to top</a></p>