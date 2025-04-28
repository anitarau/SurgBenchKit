 # added by Anita Rau April 2025

from vlmeval.smp import *
from sharedeval import precision, jaccard, recall, f1, map_for_classification, mloc_iou, f1max, f1max_thres, accuracy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sharedeval.data.cholec_helpers import *

import torch
import os
from difflib import get_close_matches 
device = "cuda" if torch.cuda.is_available() else "cpu"


def infer_data(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    """
    Predicts and evaluates data, while eval_data only evaluates
    """
    # Different models have different attributes:
    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]
    else:
        model_name = model.name
    if 'paligemma' in model_name:
        # paligemma needs its own inference
        return infer_data_paligemma(model, work_dir, name, dataset, task, prompt, **kwargs)
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    # save prompt as txt
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        if isinstance(prompt, list):
            f.write(','.join(prompt)) # for few shot
        else:
            f.write(prompt)

    # choose zero-shot or few-shot prompt
    if 'fewshot' in task['name']:
        def eval_model(frame, prompt):
            return model.generate(prompt + [frame, 'output: '] )
    else:
        def eval_model(frame, prompt):
            return model.generate([frame, prompt])

    # as API calls often fail, we save all predictions as json files, and then read those in an eval loop
    for frame, label in tqdm(dataset):
        out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
        if osp.exists(out_file) and not kwargs['override_outputs']: #TODO double check this
            continue
        else:
            if osp.exists(out_file):
                os.remove(out_file)
            try:
                ret = eval_model(frame['path'], prompt) 
                if isinstance(ret, int):  # returns int if gemini blocks the request (e.g. image contains a lot of blood)
                    dump('Blocked: ' + str(ret), out_file)
                    continue
                else: 
                    start_index = ret.find('{')
                    end_index = ret.rfind('}')  # Use rfind to get the last occurrence of '}'

                    if start_index != -1 and end_index != -1:
                        result = ret[start_index:end_index + 1]  # Include the '}'
                    else:
                        result = "{}"  # Handle case where {} are not found # was None
                    # clean up result:
                    ret = result.strip("```").strip("json").replace("\n","").replace("False", "0").replace("True", "1").replace("false", "0").replace("true", "1")
                    pred = json.loads(ret)
                    dump(pred, out_file)

            except Exception as e:
                print('Exception: ' + str(e))
                dump('Exception: ' + str(e), out_file)
                continue

    preds, labels = eval_data(model, work_dir, name, dataset, task)
    return preds, labels


def infer_data_paligemma(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    # needs on infer loop, as no json formatting promptable
    model_name = model.name
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)

    # save prompt as txt
    with open(osp.join(write_dir, 'prompt.txt'), 'w') as f:
        if isinstance(prompt, list):
            f.write(','.join(prompt))
        else:
            f.write(prompt)

    def eval_model(frame, prompt):
        return model.generate([frame, prompt])

    if "tool" in task['name'] or "cvs" in task['name'] or "heichole_action" in task['name'] or "dresden" in task['name']:
        target_length = len(dataset.labels[0][1])
        if "heichole_tool" in task['name']: #TODO needed?
            target_length = 7 # labels are padded with zeros
        elif "dresden" in task['name']:
            target_length = 11 # cut the 'null' class
        #    task.label_names = [i for i in task.label_names if not i =='null']
        for frame, label in tqdm(dataset):

            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)

                outputs = dict()
                for i, p in enumerate(prompt):
                    ret = eval_model(frame['path'], p) 
                    if ret == 'no':
                        output = 0
                    elif ret == 'yes':
                        output = 1
                    else:
                        print(" ")
                        continue
                    outputs[task.label_names[i]] = output
                if len(outputs) != target_length:
                    print('Failed on', frame['path'])
                    continue
                dump(outputs, out_file)
    

    elif 'object_detection' in task['name']:
        def process_object_detection(output):
            import re
            loc_values = re.findall(r'<loc(\d{4})>', output)
            if len(loc_values) != 4:
                return None

            y0, x0, y1, x1 = map(int, loc_values)
            return y0, x0, y1, x1
        
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)

                outputs = dict()
                for i, p in enumerate(prompt):
                    ret = eval_model(frame['path'], p)
                    num_found_objs = len(ret.split(';'))
                    for o, found_object in enumerate(ret.split(';')):
                        output = process_object_detection(found_object)
                        if output is None:
                            continue
                        if task.label_names[i] in ['tool', 'hand', 'forceps', 'needledriver', 'bovie']:  # endoscapes: only tool can have more than one instance. avos: all can be > 1
                            outputs[task.label_names[i]+str(o+1)] = list(output)
                        else:
                            outputs[task.label_names[i]] = list(output)
                dump(outputs, out_file)


    elif 'phase' in task['name']:
        if 'heichole' in task['name'] or 'cholec80' in task['name']:
            options = ['preparation', 'calot triangle dissection', 'clipping cutting', 'gallbladder dissection', 'gallbladder packaging', 'cleaning coagulation', 'gallbladder retraction']
        elif 'multibypass' in task['name']:
            options = ['preparation', 'gastric pouch creation', 'omentum division', 'gastrojejunal anastomosis', 'anastomosis test', 'jejunal separation', 'petersen space closure', 
                       'jejunojejunal anastomosis', 'mesenteric defect closure', 'cleaning & coagulation', 'disassembling', 'other intervention']
        else:
            raise NotImplementedError

        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = dict()
                    ret = eval_model(frame['path'], prompt).strip().lower() 
                    closest_match = get_close_matches(ret, options, n=1)
                    outputs['phase'] = options.index(closest_match[0])
                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')

    elif 'avos_action' in task['name']:
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = dict()
                    ret = eval_model(frame['path'], prompt).strip().lower() 
                    options = ['cutting', 'tying knots', 'suturing', 'background task']
                    closest_match = get_close_matches(ret, options, n=1)
                    outputs['action'] = options.index(closest_match[0])
                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')
    elif "triplet" in task['name']:
        for frame, label in tqdm(dataset):
            out_file = osp.join(write_dir, f'{"-".join(frame["path"].split("/")[-3:])}.json')
            if osp.exists(out_file) and not kwargs['override_outputs']:
                continue
            else:
                if osp.exists(out_file):
                    os.remove(out_file)
                try:
                    outputs = {"instrument": [], "verb": [], "target": []}
                    ret = eval_model(frame['path'], 'Which instruments are present in this image? Choose all that apply from this list: grasper, bipolar, hook, scissors, clipper, irrigator, specimen bag, none')                   
                    if not 'one' in ret:
                        tools = ret.split(',')
                        if len(tools) > 1:
                            print('multiple tools detected')
                        for i, tool in enumerate(tools):
                            tool = tool.lower()
                            if not tool in ['grasper', 'bipolar', 'hook', 'scissors', 'clipper', 'irrigator', 'specimen bag']:
                                continue
                            #Return a  dict using this JSON schema: {"instrument": [tool1,...], "verb": [activity1,...], "target": [tissue1,...]}
                            p = f'What is the {tool} doing in this image? Choose one action from these options: grasp, retract, dissect, coagulate, clip, cut, aspirate, irrigate, pack, nothing.'
                            verb = eval_model(frame['path'], p)
                            if not verb in ['grasp', 'retract', 'dissect', 'coagulate', 'clip', 'cut', 'aspirate', 'irrigate', 'pack']:
                                verb = 'null'
                                subject = 'null'
                            else:
                                p = f'What is the {tool} {verb}ing? Choose one tissue from these options: gallbladder, cystic plate, cystic duct, cystic artery, cystic pedicle, blood vessel, fluid, abdominal wall cavity, liver, adhesion, omentum, peritoneum, gut, specimen bag, nothing.'
                                subject = eval_model(frame['path'], p)
                                if not subject in ['gallbladder', 'cystic plate', 'cystic duct', 'cystic artery', 'cystic pedicle', 'blood vessel', 'fluid', 'abdominal wall cavity', 'liver', 'adhesion', 'omentum', 'peritoneum', 'gut', 'specimen bag']:
                                    subject = 'null'
                            outputs["instrument"].append(tools[i])
                            outputs["verb"].append(verb)
                            outputs["target"].append(subject)

                    dump(outputs, out_file)

                except Exception as e:
                    dump('Exception: ' + str(e), out_file)
                    print('exception')
    else:
        raise NotImplementedError
    preds, labels = eval_data(model, work_dir, name, dataset, task)
    return preds, labels


def infer_data_contrastive(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    prompt, 
    **kwargs
):
    # Predict using CLIP, OpenCLIP and SurgVLP
    preds = []
    labels = []
    if 'cholec80' in task['name'] or 'heichole_tool' in task['name'] or 'heichole_phase' in task['name'] or 'cvs' in task['name'] or 'heichole_action' in task['name'] or 'multibypass' in task['name'] or 'dresden' in task['name'] or 'disease_severity' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
    elif 'cholect45_triplet' in task['name']:
        triplet_file = osp.join(dataset.data_dir, 'dict/triplet.txt')
        label_map = {}
        with open(triplet_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                label_map[int(line[0])] = line[1]
    elif 'avos_action' in task['name']:
        label_map = {k: v for k, v in dataset.map.items()}
    else:
        raise NotImplementedError
    
    if isinstance(model.model, str):
        model_name = model.model
    elif isinstance(model.name, str):
        model_name = model.name
    else:
        model_name = model.model_path.strip('/')#VLM not API handled differently n VLMEvalKit

    for frame, label in tqdm(dataset):

        probabilities = model(prompt, frame['path'])
        if task['clip_eval_mode'] == 'singlelabel':
            pred = np.argmax(probabilities)
        elif task['clip_eval_mode'] == 'sigmoid':
            pred = probabilities[0] # remove extra dimension. Best threshold is chosen at evaluation time
        else:
            raise ValueError('Invalid eval_type')
        if task['name'] == 'cholect45_triplet_recognition':  # multimodal binary classification task, so null class is never positive, it would just be a zero vector
            pred = pred[:-1]  # remove null instrument
        elif task['name'] == 'heichole_tool_recognition':
            label = label[:len(pred)]  # GT is padded with zeros, so cut them.
        elif 'avos_action' in task['name']:
            label = label_map[label]
        labels.append(label)
        preds.append(pred)

    preds = np.array(preds)
    labels = np.array(labels)
    # write preds and labels to file
    write_dir = osp.join(work_dir, task['name'], model_name, name)
    if not osp.exists(write_dir):
        os.makedirs(write_dir)
    np.savez(osp.join(write_dir, 'results.npz'), labels=labels, preds=preds)

    eval_data_contrastive(model, work_dir, name, dataset, task, prompt, **kwargs)
    return preds, labels


def eval_data_contrastive(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    *args, 
    **kwargs
):
    # Eval for CLIP, OpenCLIP, and SurgVLP
    if isinstance(model.model, str):
        model_name = model.model
    elif isinstance(model.name, str):
        model_name = model.name
    else:
        model_name = model.model_path.strip('/')  # VLM not API handled differently n VLMEvalKit
    read_dir = osp.join(work_dir, task['name'], model_name, name)

    if 'cholec80' in task['name'] or 'heichole_tool' in task['name'] or 'heichole_phase' in task['name'] or 'cvs' in task['name'] or 'action' in task['name'] or 'multibypass' in task['name'] or 'dresden' in task['name'] or 'disease' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
    elif 'cholect45_triplet' in task['name']:
        triplet_file = osp.join(dataset.data_dir, 'dict/triplet.txt')
        label_map = {}
        with open(triplet_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split(':')
                label_map[int(line[0])] = line[1]
    else:
        raise NotImplementedError

    data = np.load(osp.join(read_dir, 'results.npz'))
    preds = data['preds']
    labels = data['labels']
    if 'dresden' in task['name']:
        labels = np.concatenate((labels[:,:10], labels[:,-1].reshape(-1,1)), axis=1)  # cut the 'null' class, which has second to last position
    if task['clip_eval_mode'] == 'sigmoid':
        if 'cholect45_triplet' in task['name']:
            threshold = f1max_thres(labels, preds)
            verbs_pred, targets_pred, instruments_pred = [], [], []
            verbs_gt, targets_gt, instruments_gt = [], [], []
            for pred in preds:
                verbs = np.zeros(len(dataset.verb_map))
                targets = np.zeros(len(dataset.target_map))
                instruments = np.zeros(len(dataset.instrument_map))
                # in pred vector get indices of 1s
                idx = np.where(pred > threshold)[0]  #get threshold from max f1 before splitting up predicitons into s,v,t
                for i in idx:
                    i_idx, v_idx, t_idx = dataset.maps[i]
                    verbs[v_idx] = 1
                    targets[t_idx] = 1
                    instruments[i_idx] = 1
                verbs_pred.append(verbs)
                targets_pred.append(targets)
                instruments_pred.append(instruments)
            for label in labels:
                verbs = np.zeros(len(dataset.verb_map))
                targets = np.zeros(len(dataset.target_map))
                instruments = np.zeros(len(dataset.instrument_map))
                idx = np.where(label == 1)[0]
                for i in idx:
                    i_idx, v_idx, t_idx = dataset.maps[i]
                    verbs[v_idx] = 1
                    targets[t_idx] = 1
                    instruments[i_idx] = 1
                verbs_gt.append(verbs)
                targets_gt.append(targets)
                instruments_gt.append(instruments)


            preds = [np.array(instruments_pred), np.array(verbs_pred), np.array(targets_pred)]
            labels = [np.array(instruments_gt), np.array(verbs_gt), np.array(targets_gt)]
            ### compute and display metrics
            print('Instruments:')
            map = map_for_classification(preds[0], labels[0])
            sliced_dict = dataset.instrument_map
            sliced_dict[6] = sliced_dict[-1]  # for some reason in the dataset the null class has key -1 for instruments but not verb and target
            del sliced_dict[-1]
            task['name'] = task['name'] + '_instrument'
            eval_metrics(labels[0], preds[0], sliced_dict, work_dir, task, model_name, name, len(labels[0]))
            print('Verb:')
            map = map_for_classification(preds[1], labels[1])
            task['name'] = task['name'].replace('_instrument', '_verb')
            eval_metrics(labels[1], preds[1], dataset.verb_map, work_dir, task, model_name, name, len(labels[1]))
            print('Target:')
            map = map_for_classification(preds[2], labels[2])
            task['name'] = task['name'].replace('_verb', '_target')
            eval_metrics(labels[2], preds[2], dataset.target_map, work_dir, task, model_name, name, len(labels[2]))
            return preds, labels
            ####################

        else:
            map = map_for_classification(labels, preds)
            threshold = f1max_thres(labels, preds)  # here it's one threshold overall, so resulting F1 is worse, but this seems more reasonable
            eval_metrics(labels, preds > threshold, label_map, work_dir, task, model_name, name, len(preds))
    elif task['clip_eval_mode'] == 'singlelabel' or task['clip_eval_mode'] == 'negative_examples':
        eval_metrics(labels, preds, label_map, work_dir, task, model_name, name, len(preds))


def eval_data(
    model, 
    work_dir, 
    name, 
    dataset, 
    task, 
    *args, 
    **kwargs
):
    # This function reads all files in a dataset and sets missing predictions to a defualt prediction.

    if isinstance(model.model, str):
        model_name = model.model
    elif hasattr(model, 'model_path'):
        model_name = model.model_path.split('/')[1]#VLM not API handled differently n VLMEvalKit
    else:
        model_name = model.name
    read_dir = osp.join(work_dir, task['name'], model_name, name)
    preds, labels = [], []
    successful_preds = 0
    evaluation_files = []
    all_files_labels = dict()
    all_files = []
    for frame, label in dataset.labels:
        out_file = osp.join(read_dir, f'{"-".join(frame.split("/")[-3:])}.json')
        all_files.append(out_file)
        all_files_labels[out_file] = label
        if osp.exists(out_file):
            evaluation_files.append(out_file)


    if 'dresden_anatomy' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.zeros(len(label_map)-1))  # -1 for the 'null' class we ignore
        for file in tqdm(all_files):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    preds.append(default)
                elif len(pred) != len(default):
                    preds.append(default)
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1

            label = all_files_labels[file]
            label_fixed = np.delete(label, -2)  # cut null class
            labels.append(label_fixed)

    elif 'multibypass140' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        for file in tqdm(all_files):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    preds.append(default)
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1

            label = all_files_labels[file]
            labels.append(label)
        labels = np.array(labels).reshape(-1,1)

    elif 'cholec80' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        for file in tqdm(all_files):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    preds.append(default) 
                elif len(pred) != len(label_map) and 'phase' not in task['name']:
                    preds.append(default) 
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1
            label = all_files_labels[file]
            labels.append(label)
        if 'cholec80_phase_recognition' in task['name']:
            labels = np.array(labels).reshape(-1,1)
    
    elif 'avos_action' in task['name']: # TODO is this needed?
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        label_map_inversed = {label: idx for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(3))  # 3 is background
        for file in all_files:  # os.listdir(read_dir):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
            # check if pred is a dict
                if not isinstance(pred, dict):
                    preds.append(default)
                elif "Blocked" in pred or "Exception" in pred:
                    preds.append(default)
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1
            folder = file.split('-')[2]
            frame = file.split('-')[3].strip('.json')
            label = label_map_inversed[all_files_labels[file]]
            labels.append(label)
        labels = np.array(labels).reshape(-1,1)
        print(' ')


    elif 'cholect45_triplet' in task['name']:
        label_map = {}
        offset = 0
        for d in [dataset.instrument_map, dataset.verb_map, dataset.target_map]:
            for key, value in d.items():
                if not value == 'null_instrument':
                    label_map[offset] = value
                    offset += 1
        verb_labels = get_triplet_component_labels(dataset.data_dir +'/verb')
        target_labels = get_triplet_component_labels(dataset.data_dir + '/target')
        instrument_labels = get_triplet_component_labels(dataset.data_dir + '/instrument')
        verbs, targets, instruments = [], [], []
        verbs_gt, targets_gt, instruments_gt = [], [], []
        default = {"instrument": ["null"], "verb": ["null"], "target": ["null"]}
        for file in tqdm(all_files):
            if file not in evaluation_files:
                pred = default
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                    if "Blocked" in pred:
                        pred = default
                    elif "Exception" in pred:
                        pred = default
                    else:
                        successful_preds += 1
            verb_logits, target_logits, instrument_logits = pred_to_logits(pred, dataset)
            verbs.append(verb_logits)
            targets.append(target_logits)
            instruments.append(instrument_logits)
            file = file.split('/')[-1]
            if file[0] == 'r':
                folder = file.split('-')[1]
                frame = str(int(file.split('-')[2].split('.')[0]))
            else:
                folder = file.split('-')[0]
                frame = str(int(file.split('-')[1].split('.')[0]))
            verb_logits_gt, target_logits_gt, instrument_logits_gt = verb_labels[folder + '-' + frame], target_labels[folder + '-' + frame], instrument_labels[folder + '-' + frame]
            verbs_gt.append(verb_logits_gt)
            targets_gt.append(target_logits_gt)
            instruments_gt.append(instrument_logits_gt)

        preds = [np.array(instruments), np.array(verbs), np.array(targets)]
        labels = [np.array(instruments_gt), np.array(verbs_gt), np.array(targets_gt)]

    elif 'heichole' in task['name'] and 'skill_assessment' not in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)
        if 'heichole_tool_recognition' in task['name']:
                default = default[:7]
        for file in tqdm(all_files):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                     preds.append(default) 
                elif len(pred) != len(label_map) and 'phase' not in task['name']:
                    preds.append(default) 
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1
            label = all_files_labels[file]
            # HeiChole dataset states "Tools 7-20 Reserved for future additions" --> remove
            if 'heichole_tool_recognition' in task['name']:
                label = label[:7]
            labels.append(label)

    elif 'cvs' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        default = np.atleast_1d(np.array(dataset.labels[0][1]) * 0)

        for file in tqdm(all_files):
            if file not in evaluation_files:
                preds.append(default)
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    preds.append(default)
                elif len(pred) != 3:
                    preds.append(default)
                else:
                    preds.append(np.array(list(pred.values())) * 1)
                    successful_preds += 1
            label = all_files_labels[file]
            labels.append(label)
    
    elif 'endoscapes_object_detection' in task['name']: # TODO is this needed?
        label_map_inverted = dataset.category_ids_to_name
        label_map = {v: k for k, v in label_map_inverted.items()}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        cocoGt = COCO(osp.join(dataset.data_dir, dataset.split, 'annotation_coco.json'))
        image_ids_to_evaluate = []
        default = {}
        label_counts = {label: 0 for label in label_map.keys()}
        for file in all_files:
            if file not in evaluation_files:
                pred = default
            else:
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    pred = default
                else:
                    successful_preds += 1
            folder = file.split('/')[-1].split('-')[2].split('_')[0]
            frame = file.split('/')[-1].split('_')[1].split('.')[0] + '.jpg'
            label = all_files_labels[file] # labels_dict[osp.join(dataset.data_dir, dataset.split, folder + '_' + frame)]
            image_id =  dataset.file_names_to_id[folder + '_' + frame]
            image_ids_to_evaluate.append(image_id)
            # add to label_counts
            for category_name in label.keys():
                if not category_name in label_map:
                    continue
                label_counts[category_name] += len(label[category_name])

            for category_name in pred.keys():
                if len(pred[category_name]) == 4:
                    y0, x0, y1, x1 = pred[category_name]
                else:
                    continue
                if 'gemini' in model_name:
                    x0 *= label['im_size_wh'][0] / 1000  #TODO this is for Gemini. Check how other models handle this
                    x1 *= label['im_size_wh'][0] / 1000
                    y0 *= label['im_size_wh'][1] / 1000
                    y1 *= label['im_size_wh'][1] / 1000
                elif 'paligemma' in model_name: # paligemma assumes coodinates for 1024 X 1024 images. endoscapes images are 854 x 480
                    x0 *= 854 / 1024
                    x1 *= 854 / 1024
                    y0 *= 480 / 1024
                    y1 *= 480 / 1024
                width = x1 - x0
                height = y1 - y0
                if not category_name.replace(' ','_') in label_map:
                    continue
                if not 'tool' in category_name:
                    category_id = label_map[category_name.replace('cyctic', 'cystic').replace(' ', '_')]
                    coco_format = {"image_id": image_id, "category_id": category_id, "bbox": [x0, y0, width, height], "score": 1.0}
                else:
                    coco_format = {"image_id": image_id, "category_id": label_map['tool'], "bbox": [x0, y0, width, height], "score": 1.0}
                preds.append(coco_format)
        dump(preds, osp.join(read_dir, 'results.json'))
        cocoDt = cocoGt.loadRes(osp.join(read_dir, 'results.json'))
        image_ids_to_evaluate = list(set(image_ids_to_evaluate))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = image_ids_to_evaluate
        results = []
        if dataset.category == 'all':
            for category_id, category_name in dataset.category_ids_to_name.items():
                print('#####################################   Evaluating category:', category_name)
                cocoEval.params.catIds = [category_id]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # Extracting metrics
                AP_50_95_all = cocoEval.stats[0]  # AP@0.50:0.95 for area=all
                AP_50_all = cocoEval.stats[1]     # AP@0.50 for area=all
                AP_75_all = cocoEval.stats[2]     # AP@0.75 for area=all
                AR_1_all = cocoEval.stats[6]      # AR@1 for area=all
                AR_10_all = cocoEval.stats[7]     # AR@10 for area=all
                
                # Storing results for the category
                results.append({
                    'Class': category_name,
                    'AP@0.50:0.95': AP_50_95_all,
                    'AP@0.50': AP_50_all,
                    'AP@0.75': AP_75_all,
                    'AR@1': AR_1_all,
                    'AR@10': AR_10_all,
                })
        df = pd.DataFrame(results)

        # Separate "tool" and "anatomies"
        tool_df = df[df['Class'] == 'tool']
        anatomies_df = df[df['Class'] != 'tool']

        def weighted_average(df):
            metrics = df.select_dtypes(include=['float64']).columns
            total_weight = sum(label_counts.get(row['Class'], 0) for _, row in df.iterrows())
            weighted_metrics = {
                metric: sum(row[metric] * label_counts.get(row['Class'], 0) for _, row in df.iterrows()) / total_weight
                for metric in metrics
            }
            return weighted_metrics
        
        tool_weighted_avg = weighted_average(tool_df)
        tool_weighted_avg['Class'] = 'Weighted Average'

        anatomies_weighted_avg = weighted_average(anatomies_df)
        anatomies_weighted_avg['Class'] = 'Weighted Average'

        # Calculate average metrics
        tool_avg = tool_df.mean(numeric_only=True).to_dict()
        anatomies_avg = anatomies_df.mean(numeric_only=True).to_dict()

        # Add "Average" row
        tool_avg['Class'] = 'Average'
        tool_avg['Successful Preds'] = successful_preds  
        anatomies_avg['Class'] = 'Average'
        anatomies_avg['Successful Preds'] = successful_preds
        tool_df = pd.concat([tool_df, pd.DataFrame([tool_weighted_avg]), pd.DataFrame([tool_avg])], ignore_index=True).round(2)
        anatomies_df = pd.concat([anatomies_df, pd.DataFrame([anatomies_weighted_avg]), pd.DataFrame([anatomies_avg])], ignore_index=True).round(2)

        # Save to CSV
        if 'fewshot' in task['name']:
            tool_df.to_csv(work_dir + 'metrics_endoscapes_tool_detection_fewshot_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
            anatomies_df.to_csv(work_dir + 'metrics_endoscapes_anatomy_detection_fewshot_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        else: 
            tool_df.to_csv(work_dir + 'metrics_endoscapes_tool_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
            anatomies_df.to_csv(work_dir + 'metrics_endoscapes_anatomy_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)

        return preds, labels

    elif 'avos_object_detection' in task['name']:  # TODO is this needed?
        label_map_inverted = dataset.category_ids_to_name
        label_map = {v: k for k, v in label_map_inverted.items()}
        labels_dict = {filename: label_array for filename, label_array in dataset.labels}
        cocoGt = COCO('avos_coco_annotations.json')
        image_ids_to_evaluate = []
        default = {}
        label_counts = {label: 0 for label in label_map.keys()}
        frame_name_idx = 4
        if 'paligemma' in model_name:
            frame_name_idx = 5
        for file in tqdm(all_files):
            if file not in evaluation_files:
                pred = default
            else: 
                with open(file, 'r') as f:
                    pred = json.load(f)
                if "Blocked" in pred or "Exception" in pred:
                    pred = default
                else:
                    successful_preds += 1
            parts = file.split('-')
            frame = '-'.join(parts[frame_name_idx:]).split('.')[0] + '.jpg'
            label = all_files_labels[file] # labels_dict[osp.join(dataset.data_dir, dataset.split, folder + '_' + frame)]
            # add to label_counts
            for category_name in label.keys():
                if not category_name in label_map:
                    continue
                label_counts[category_name] += len(label[category_name])

            for category_name in pred.keys():
                if len(pred[category_name]) == 4:
                    y0, x0, y1, x1 = pred[category_name]
                    if isinstance(y0, dict):
                        continue
                    if isinstance(y0, str):
                        y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
                else:
                    continue
                image_id =  dataset.file_names_to_id[frame]
                image_ids_to_evaluate.append(image_id)
                x0 *= label['im_size_wh'][0] / 1000  #TODO this is for Gemini. Check how other models handle this
                x1 *= label['im_size_wh'][0] / 1000
                y0 *= label['im_size_wh'][1] / 1000
                y1 *= label['im_size_wh'][1] / 1000
                width = x1 - x0
                height = y1 - y0
                if 'forceps' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['forceps'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'bovie' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['bovie'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'needledriver' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['needledriver'], "bbox": [x0, y0, width, height], "score": 1.0}
                elif 'hand' in category_name:
                    coco_format = {"image_id": image_id, "category_id": label_map['hand'], "bbox": [x0, y0, width, height], "score": 1.0}
                preds.append(coco_format)
        dump(preds, osp.join(read_dir, 'results.json'))
        cocoDt = cocoGt.loadRes(osp.join(read_dir, 'results.json'))
        image_ids_to_evaluate = list(set(image_ids_to_evaluate))
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        cocoEval.params.imgIds = image_ids_to_evaluate
        results = []
        if dataset.category == 'all':
            for category_id, category_name in dataset.category_ids_to_name.items():
                print('#####################################   Evaluating category:', category_name)
                cocoEval.params.catIds = [category_id]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                # Extracting metrics
                AP_50_95_all = cocoEval.stats[0]  # AP@0.50:0.95 for area=all
                AP_50_all = cocoEval.stats[1]     # AP@0.50 for area=all
                AP_75_all = cocoEval.stats[2]     # AP@0.75 for area=all
                AR_1_all = cocoEval.stats[6]      # AR@1 for area=all
                AR_10_all = cocoEval.stats[7]     # AR@10 for area=all
                
                # Storing results for the category
                results.append({
                    'Class': category_name,
                    'AP@0.50:0.95': AP_50_95_all,
                    'AP@0.50': AP_50_all,
                    'AP@0.75': AP_75_all,
                    'AR@1': AR_1_all,
                    'AR@10': AR_10_all,
                })
        df = pd.DataFrame(results)

        # Separate "tool" and "anatomies"
        hand_df = df[df['Class'] == 'hand']
        tools_df = df[df['Class'] != 'hand']

        def weighted_average(df):
            metrics = df.select_dtypes(include=['float64']).columns
            total_weight = sum(label_counts.get(row['Class'], 0) for _, row in df.iterrows())
            weighted_metrics = {
                metric: sum(row[metric] * label_counts.get(row['Class'], 0) for _, row in df.iterrows()) / total_weight
                for metric in metrics
            }
            return weighted_metrics
        
        hand_weighted_avg = weighted_average(hand_df)
        hand_weighted_avg['Class'] = 'Weighted Average'

        tools_weighted_avg = weighted_average(tools_df)
        tools_weighted_avg['Class'] = 'Weighted Average'

        # Calculate average metrics
        hand_avg = hand_df.mean(numeric_only=True).to_dict()
        tools_avg = tools_df.mean(numeric_only=True).to_dict()

        # Add "Average" row
        hand_avg['Class'] = 'Average'
        hand_avg['Successful Preds'] = successful_preds  
        tools_avg['Class'] = 'Average'
        tools_avg['Successful Preds'] = successful_preds
        hand_df = pd.concat([hand_df, pd.DataFrame([hand_weighted_avg]), pd.DataFrame([hand_avg])], ignore_index=True).round(2)
        tools_df = pd.concat([tools_df, pd.DataFrame([tools_weighted_avg]), pd.DataFrame([tools_avg])], ignore_index=True).round(2)

        # Save to CSV
        hand_df.to_csv(work_dir + 'metrics_avos_hand_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        tools_df.to_csv(work_dir + 'metrics_avos_tools_detection_' + model_name.replace('/','') + '_' + name + '.csv', index=False)

        return preds, labels
    
    elif 'jigsaws' in task['name'] or 'autolaparo' in task['name'] or 'heichole_skill_assessment' in task['name']:
        label_map = {idx: label for idx, label in enumerate(task['label_names'])}
        successful_preds = 0
        for i, pred in enumerate(preds):
            pattern = '|'.join([re.escape(label) for label in task.label_names])
            matches = re.findall(pattern, pred)
            if len(matches) == 1:
                preds[i] = matches[0]
                successful_preds += 1
            elif len(matches) == 0:
                preds[i] = task.label_names[0]
            else:
                preds[i] = matches[-1]
                successful_preds += 1
         preds = np.array(preds)
        labels = np.array(labels)


    ### compute and display metrics
    if 'triplet' in task['name']:
        print('Instruments:')
        map = map_for_classification(preds[0], labels[0])
        sliced_dict = {k: label_map[k] for k in range(preds[0].shape[1])}
        task['name'] = task['name'] + '_instrument'
        eval_metrics(labels[0], preds[0], sliced_dict, work_dir, task, model_name, name, successful_preds)
        print('Verb:')
        map = map_for_classification(preds[1], labels[1])
        sliced_dict = {k-preds[0].shape[1]: label_map[k] for k in range(preds[0].shape[1], preds[1].shape[1]+ preds[0].shape[1])}
        task['name'] = task['name'].replace('_instrument', '_verb')
        eval_metrics(labels[1], preds[1], sliced_dict, work_dir, task, model_name, name, successful_preds)
        print('Target:')
        map = map_for_classification(preds[2], labels[2])
        sliced_dict = {k-preds[0].shape[1]-preds[1].shape[1]: label_map[k] for k in range(preds[0].shape[1] + preds[1].shape[1], preds[2].shape[1] + preds[0].shape[1] + preds[1].shape[1])}
        task['name'] = task['name'].replace('_verb', '_target')
        eval_metrics(labels[2], preds[2], sliced_dict, work_dir, task, model_name, name, successful_preds)
        return preds, labels

    if not 'error_detection' in task['name']:
        preds = np.array(preds, dtype=np.uint8)
        labels = np.array(labels, dtype=np.uint8)
    if len(labels) != len(all_files): #TODO needed? not sure what this does
        raise NotImplementedError 

    eval_metrics(labels, preds, label_map, work_dir, task, model_name, name, successful_preds, len_test_files=len(all_files))

    return preds, labels


def eval_metrics(
    labels, 
    preds, 
    label_map, 
    work_dir, 
    task, 
    model_name, 
    name,
    successful_preds,
    len_test_files=None
):
    print(f'Finished evaluating {model_name}')
    if not 'phase' in task['name'] \
        and not 'avos' in task['name'] \
        and not 'error_detection' in task['name'] \
        and not 'error_classification' in task['name'] \
        and not 'disease_severity' in task['name'] \
        and not 'intermountain_skill_assessment' in task['name']:
        for i in range(labels.shape[1]):
            print(f'Instances of {label_map[i]}: {np.sum(labels[:, i])}')
        print('Total instances:', labels.shape[0])
        for i in range(preds.shape[1]):
            print(f'Predicted instances of {label_map[i]}: {np.sum(preds[:, i])}')

    if 'phase_recognition' in task['name'] or 'avos' in task['name'] or 'gesture_classification' in task['name']:
        out  = np.unique(labels, axis=0, return_counts=True)
        total_numbers_per_class = out[1]
        print('total number of labels per class: ', total_numbers_per_class)
    elif 'disease_severity' in task['name'] or 'intermountain_skill_assessment' in task['name']: # TODO is this needed?
        total_numbers_per_class = np.unique(labels, return_counts=True)[1]
        print(total_numbers_per_class)
    else:
        total_numbers_per_class = np.sum(labels, axis=0)

    if 'intermountain_skill_assessment' in task['name']: # TODO is this needed?
        f1_values = f1(labels, preds, average=None)
        print('f1_values: ', f1_values)
        accuracy_macro = accuracy(labels, preds)
        metrics_df = pd.DataFrame({
            'Class': ['Average'],
            'F1 Score': [np.mean(f1_values)],
            'Accuracy': [accuracy_macro],
            'Successful Preds': [successful_preds]
        })
        print(metrics_df.to_string(index=False))
        print('-'*60)
        metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        return preds, labels

    if 'error_detection' in task['name']: # TODO is this needed?
        # print('mIoU: ', mloc_iou(labels, preds))
        # print('-'*60)
        metrics_df = pd.DataFrame({
            'Class': ['Average'],
            'mIoU': [mloc_iou(labels, preds)],
            'Successful Preds': [successful_preds]
        })
        print(metrics_df.to_string(index=False))
        print('-'*60)
        metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
        return preds, labels

    if 'error_classification' in task['name']: # TODO is this needed?
        print('-'*100)
        print('labels: ', labels)
        print('preds: ', preds)
        print('mAcc: ', accuracy(labels.flatten(), preds.flatten()))
        print('-'*100)

    ## recall, precision, f1, jaccard
    recall_values = recall(labels, preds, average=None)
    precision_values = precision(labels, preds,average=None)
    jaccard_values = jaccard(labels, preds, average=None)
    f1_values = f1(labels, preds, average=None)
    recall_micro = recall(labels, preds, average='micro')
    precision_micro = precision(labels, preds, average='micro')
    jaccard_micro = jaccard(labels, preds, average='micro')
    f1_micro = f1(labels, preds, average='micro')
    accuracy_macro = accuracy(labels.flatten(), preds.flatten())

    # weighted average
    recall_avg = recall(labels, preds, average='weighted')
    precision_avg = precision(labels, preds, average='weighted')
    jaccard_avg = jaccard(labels, preds, average='weighted')
    f1_avg = f1(labels, preds, average='weighted')

    if 'dresden_anatomy' in task['name']: # TODO is this needed?
        ### remove null class
        null_index = next(key for key, value in label_map.items() if value == 'null')
        map_ids = [i for i in range(len(label_map)) if i != null_index]
    elif 'phase_recognition' in task['name'] or 'avos' in task['name']:
        if np.ndim(labels) != np.ndim(preds):
            if np.ndim(labels) == 1:
                labels = labels.reshape(-1,1)
            elif np.ndim(preds) == 1:
                preds = preds.reshape(-1,1)
        map_ids = np.unique([labels, preds])
    else:
        map_ids = range(len(label_map))

    # printing/saving
    print('Metrics:')
    metrics_df = pd.DataFrame({
        'Class': [f'{label_map[i]}' for i in map_ids],
        'Recall': recall_values,
        'Precision': precision_values,
        'Jaccard': jaccard_values,
        'F1 Score': f1_values
    })
    weighted_avg_df = pd.DataFrame({
        'Class': ['Weighted Average'],
        'Recall': [recall_avg],
        'Precision': [precision_avg],
        'Jaccard': [jaccard_avg],
        'F1 Score': [f1_avg]
    })
    avg_df = pd.DataFrame({
        'Class': ['Average'],
        'Recall': [np.mean(recall_values)],
        'Precision': [np.mean(precision_values)],
        'Jaccard': [np.mean(jaccard_values)],
        'F1 Score': [np.mean(f1_values)],
        'Accuracy': [accuracy_macro],
        'Successful Preds': [successful_preds]
    })
    metrics_df = pd.concat([metrics_df, weighted_avg_df, avg_df], ignore_index=True)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(work_dir + 'metrics_' + task['name'] + '_' + model_name.replace('/','') + '_' + name + '.csv', index=False)
    return preds, labels