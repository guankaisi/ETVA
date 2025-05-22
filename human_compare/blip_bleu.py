# https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/Scores_with_CLIP/Scores_with_CLIP.py#L235
import os
import torch
import cv2
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pycocoevalcap.bleu.bleu import Bleu

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
def compute_max(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)
def calculate_blip_bleu(video_path, original_text, blip2_model, blip2_processor, device='cuda'):
    # # Load the video
    cap = cv2.VideoCapture(video_path)

    # # scorer_cider = Cider()
    bleu1 = Bleu(n=1)
    bleu2 = Bleu(n=2)
    bleu3 = Bleu(n=3)
    bleu4 = Bleu(n=4)

    # # Extract frames from the video
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame,(224,224))  # Resize the frame to match the expected input size
        frames.append(resized_frame)

    # Convert numpy arrays to tensors, change dtype to float, and resize frames
    tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
    # Get five captions for one video
    # print(tensor_frames.shape)
    Num_Frames = 8
    captions = []
    # for i in range(Num):
    N = len(tensor_frames)
    indices = torch.linspace(0, N - 1, Num_Frames).long()
    extracted_frames = torch.index_select(tensor_frames, 0, indices)
    
    for i in range(Num_Frames):
        frame = extracted_frames[i]
        inputs = blip2_processor(images=frame, return_tensors="pt").to(device, torch.float16)
        generated_ids = blip2_model.generate(**inputs)
        generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        captions.append(generated_text)


    original_text = [original_text]
    
    bleu1_score = (compute_max(bleu1, original_text, captions))
    bleu2_score = (compute_max(bleu2, original_text, captions))
    bleu3_score = (compute_max(bleu3, original_text, captions))
    bleu4_score = (compute_max(bleu4, original_text, captions))

    blip_bleu_caps_avg = (bleu1_score + bleu2_score + bleu3_score + bleu4_score)/4
     
    return blip_bleu_caps_avg

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip2_processor = AutoProcessor.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)
    video_path = 'example1.mp4'
    video = cv2.VideoCapture('example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    frame_tensors = frames2tensor(frames, device=device).squeeze(0)
    print(frame_tensors.shape)
    text = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
    blip_bleu = calculate_blip_bleu(video_path, text, blip2_model, blip2_processor)
    print(type(blip_bleu))
    print(blip_bleu)