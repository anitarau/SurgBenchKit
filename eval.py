# added by Anita Rau April 2025

from vlmeval.config import supported_VLM, model_map, data_map, ShellModel
from vlmeval.prompts import get_prompts
import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    if 'CLIP' in cfg.model.name or 'SurgVLP' in cfg.model.name:
        model = supported_VLM[cfg.model.name](eval_type=cfg.task['clip_eval_mode'])
    elif 'eval' in cfg.eval_mode:
        model = ShellModel(name=cfg.model.name)  # No need to load model weights for eval only
    else:
        model = supported_VLM[cfg.model.name]()
        
    if not hasattr(model, 'name'):
        model.name = cfg.model.name
    dataset = data_map.get(cfg.task.data)(config=cfg.task, split='test')

    # Choose zero or few shot prompt
    if hasattr(cfg.task, 'shots') and cfg.task.shots != 'zero':
        shots = cfg.task.shots
        task_name = cfg.task.name.replace('few', f'{shots}')
        prompt = get_prompts(cfg.task.data_config.data_dir, task_name, cfg.model.name)
    else:
        if cfg.task.name == 'intermountain_skill_assessment':
            cfg.task.name = f'{cfg.task.name}_{cfg.task.data_config.category}'
        prompt = get_prompts(cfg.task.data_config.data_dir, cfg.task.name, cfg.model.name)

    if cfg.model.contrastive:
        cfg.eval_mode = f'{cfg.eval_mode}_contrastive'

    model_map.get(cfg.eval_mode)(model,
                                   cfg.workdir,
                                   cfg.exp_name,
                                   dataset,
                                   cfg.task,
                                   prompt,
                                   override_outputs=cfg.override_outputs
                                   )

if __name__ == '__main__':
    main()
