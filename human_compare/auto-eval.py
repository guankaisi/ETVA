import argparse
import cv2
from tqdm import tqdm
import json
import numpy as np
import os
import gc
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from viclip_score import ViCLIP_Score, CLIP_setup
from clip_score import Clip_Score
from umt_score import UMTScore, ModelConfig
from blip_bleu import calculate_blip_bleu
from blip_rouge import calculate_blip_rouge
from video_score import VideoScore, load_video_score_model
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def normalize(data):
    return (data/255.0-v_mean)/v_std
def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert(len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:,:,::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    # vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    vid_tube = torch.from_numpy(vid_tube).float()
    return vid_tube.to(device)
def process_video(video_path, model):
    video_path = os.path.join(video_path, model)
    video_path_list =  [path for path in os.listdir(video_path) if path.endswith('.mp4')]
    # sort by number
    video_path_list.sort(key=lambda x: int(x.split('.')[0]))
    video_list = [os.path.join(video_path, video) for video in video_path_list]
    # print(video_path_list)
    video_frame_list = []
    print("Processing video frames...")
    for video in tqdm(video_list):
        video_frame = []
        video = cv2.VideoCapture(video)
        for frame in _frame_from_video(video):
            video_frame.append(frame)
        video_frame_list.append(video_frame)
    return video_frame_list, video_list
    
    

def Calculate_Metric(model, prompts_list, args):
    metric = args.metric
    video_frame_list, video_list = process_video(args.video_path, model)
    if metric == 'viclip_score':
        outpath = os.path.join(args.outpath, model)
        f = open(f'{outpath}_viclip_score.json', 'a+')
        clip, tokenizer = CLIP_setup('viclip')
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            json_new = {}
            video_frames = video_frame_list[i]
            frames_tensor = frames2tensor(video_frames)
            score = ViCLIP_Score(frames_tensor, prompt, clip, tokenizer)
            json_new['prompt'] = prompt
            json_new['viclip_score'] = str(score)
            f.write(json.dumps(json_new)+'\n')
        f.close()
            
    if metric == 'umt_score':
        outpath = os.path.join(args.outpath, 'umt')
        f = open(f'{outpath}/{model}_umt_score.json', 'a+')
        model_config = ModelConfig()
        device = torch.device('cuda')
        umt_model = UMTScore(model_config)
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            video_frames = video_frame_list[i]
            frames_tensor = frames2tensor(video_frames)
            score = umt_model.calculate_umt_score(frames_tensor, prompt)
            js_new = {}
            js_new['prompt'] = prompt
            js_new['umt_score'] = str(score)
            f.write(json.dumps(js_new)+'\n')
        f.close()
    if metric == 'clip_score':
        outpath = os.path.join(args.outpath, 'clip')
        f = open(f'{outpath}/{model}_clip_score.json', 'a+')
        Clip = Clip_Score()
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            video_frames = video_frame_list[i]
            frames_tensor = frames2tensor(video_frames)
            score = Clip.calculate_clip_score(frames_tensor, prompt)
            js_new = {}
            js_new['prompt'] = prompt
            js_new['clip_score'] = str(score)
            f.write(json.dumps(js_new)+'\n')
        f.close()
        torch.cuda.empty_cache()
    if metric == 'videoscore':
        outpath = os.path.join(args.outpath, 'videoscore')
        videoscore_model, processor = load_video_score_model('/fs/fast/share/aimind_files/video_eval/models/VideoScore-Qwen2-VL')
        f = open(f'{outpath}/{model}_videoscore.json', 'a+')
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            video_path = video_list[i]
            score = VideoScore(videoscore_model, processor, video_path, prompt)
            js_new = {}
            js_new['prompt'] = prompt
            js_new['videoscore'] = score
            f.write(json.dumps(js_new)+'\n')
            torch.cuda.empty_cache()
            gc.collect()
        f.close()
    if metric == 'blip_bleu':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip2_processor = AutoProcessor.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
        outpath = os.path.join(args.outpath, 'blip_bleu')
        f = open(f'{outpath}/{model}_blip_bleu.json', 'a+')
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            video_path = video_list[i]
            score = calculate_blip_bleu(video_path, prompt, blip2_model, blip2_processor)
            js_new = {}
            js_new['prompt'] = prompt
            js_new['blip_bleu'] = score
            f.write(json.dumps(js_new)+'\n')
        f.close()
    if metric == 'blip_rouge':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip2_processor = AutoProcessor.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b")
        blip2_model = Blip2ForConditionalGeneration.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
        outpath = os.path.join(args.outpath, 'blip_rouge')
        f = open(f'{outpath}/{model}_blip_rouge.json', 'a+')
        for i in tqdm(range(len(prompts_list))):
            prompt = prompts_list[i]
            video_path = video_list[i]
            score = calculate_blip_rouge(video_path, prompt, blip2_model, blip2_processor)
            js_new = {}
            js_new['prompt'] = prompt
            js_new['blip_rouge'] = score
            f.write(json.dumps(js_new)+'\n')
        f.close()
            
            
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--video_path', type=str, default='/fs/fast/share/aimind_files/human_eval/static/video_folder/')
    args.add_argument('--prompt_path', type=str, default='../prompt_small/prompts_small.json')
    # model-name is a list of model names
    args.add_argument('--model_names', type=str, nargs='+', default=['sora', 'kling'])
    args.add_argument('--metric', type=str, default='viclip')
    args.add_argument('--outpath', type=str, default='/fs/fast/share/aimind_files/video_eval/human_compare/result')
    args = args.parse_args()
    model_list = args.model_names

    # load prompts list 
    with open(args.prompt_path, 'r') as f:
        json_list = [json.loads(line) for line in f.readlines()]
        prompts_list = [json_list[i]['prompt'] for i in range(len(json_list))]

    for model in model_list:
        print(f"Model: {model}")
        result_score = Calculate_Metric(model, prompts_list, args)
        

