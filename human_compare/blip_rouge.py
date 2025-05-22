import os
import torch
import cv2
import numpy as np
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from pycocoevalcap.rouge.rouge import Rouge

def compute_max_rouge(scorer, gt_prompts, pred_prompts):
    scores = []
    for pred_prompt in pred_prompts:
        for gt_prompt in gt_prompts:
            cand = {0: [pred_prompt]}
            ref = {0: [gt_prompt]}
            score, _ = scorer.compute_score(ref, cand)
            scores.append(score)
    return np.max(scores)

def calculate_blip_rouge(video_path, original_text, blip2_model, blip2_processor, device='cuda'):
    cap = cv2.VideoCapture(video_path)

    rouge_scorer = Rouge()
    
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        resized_frame = cv2.resize(frame, (224, 224))  # Resize to match the input size
        frames.append(resized_frame)

    tensor_frames = torch.stack([torch.from_numpy(frame).permute(2, 0, 1).float() for frame in frames])
    
    Num_Frames = 8
    captions = []
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
    rouge_score = compute_max_rouge(rouge_scorer, original_text, captions)
    
    return rouge_score

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip2_processor = AutoProcessor.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b")
    blip2_model = Blip2ForConditionalGeneration.from_pretrained("/fs/fast/share/aimind_files/video_eval/models/blip2-opt-2.7b", torch_dtype=torch.float16).to(device)

    video_path = 'example1.mp4'
    text = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'

    blip_rouge = calculate_blip_rouge(video_path, text, blip2_model, blip2_processor)
    print(type(blip_rouge))
    print(blip_rouge)