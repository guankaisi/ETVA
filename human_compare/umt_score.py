from third_party.umt.umt import UMT
import cv2
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Any
from transformers import BertTokenizer
# 定义 TextEncoders

@dataclass
class TextEncoderConfig:
    name: str = "bert_large"
    pretrained: str = "/fs/fast/share/aimind_files/video_eval/models/bert-large-uncased"
    config: str = "third_party/umt/bert/config_bert_large.json"
    d_model: int = 1024
    fusion_layer: int = 19

@dataclass
class VisionEncoderConfig:
    name: str = "vit_l16"
    img_size: int = 224
    patch_size: int = 16
    d_model: int = 1024
    encoder_embed_dim: int = 1024
    encoder_depth: int = 24
    encoder_num_heads: int = 16
    drop_path_rate: float = 0.3
    num_frames: int = 8
    tubelet_size: int = 1
    use_checkpoint: bool = True
    checkpoint_num: int = 24
    clip_decoder_embed_dim: int = 1024
    clip_output_dim: int = 768
    clip_return_layer: int = 0
    clip_student_return_interval: int = 1
    pretrained: str = '/fs/fast/share/aimind_files/video_eval/models/UMT/ret_msrvtt_l16_25m.pth'
    keep_temporal: bool = True

@dataclass
class ModelConfig:
    model_cls: str = "UMT"
    vision_encoder: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    multimodal: Dict[str, Any] = field(default_factory=dict)
    embed_dim: int = 768
    temp: float = 0.07

class UMTScore:
    def __init__(self, model_config, device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):
        self.model_config = model_config
        self.device = device
        self.umt_model = UMT(model_config)
        self.umt_model.to(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_config.text_encoder.pretrained)
    def calculate_umt_score(self, frames_tensor, text):
        text_tensors = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        _, text_feature = self.umt_model.encode_text(text_tensors)
        _, video_feature = self.umt_model.encode_vision(frames_tensor)
        video_feature = torch.squeeze(video_feature).mean(dim=0)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        video_feature /= video_feature.norm()
        umt_score = text_feature @ video_feature
        return umt_score.detach().cpu().numpy()[0]


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

if __name__ == "__main__":
    model_config = ModelConfig()
    tokenizer = BertTokenizer.from_pretrained('/fs/fast/share/aimind_files/video_eval/models/bert-large-uncased')
    path = '/fs/fast/share/aimind_files/video_eval/models/UMT/ret_msrvtt_l16_25m.pth'
    state_dict = torch.load(path, map_location="cuda")
    # print(len(state_dict))
    device = torch.device('cuda')
    # print(model_config.text_encoder)
    umt_model = UMT(model_config)
    umt_model.to(device)
    text_candidates = ["A man play with a dog.",
    "A dog play with a man.",
    "a fucking man play with a fucking dog.",
    "jalja ;lfjdlfj kla; fj;lfjl; jaljfalj jljklj;lj",
    "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run"]
    video = cv2.VideoCapture('example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    frames_tensor = frames2tensor(frames)
    print(frames_tensor.shape)
    text_tensors = tokenizer(text_candidates, padding=True, truncation=True, return_tensors='pt').to(device)
    x1, x2 = umt_model.encode_text(text_tensors)
    x3, x4 = umt_model.encode_vision(frames_tensor)
    x4 = torch.squeeze(x4)
    x4 = torch.mean(x4,dim=0)
    print(x1.shape, x2.shape)
    print(x3.shape, x4.shape)
    umt_score = x2 @ x4
    print(umt_score)
    label_probs = (100.0 * umt_score).softmax(dim=-1)
    probs, idxs = label_probs.topk(5, dim=-1)
    print(idxs)
    ret_texts = [text_candidates[i] for i in idxs.cpu().detach().numpy().tolist()]
    probes = probs.cpu().detach().numpy().tolist()
    for t, p in zip(ret_texts, probs):
        print(f'text: {t} ~ prob: {p:.4f}')
    # umt_model = UMTScore(model_config, device)
    # umt_score = umt_model.calculate_umt_score(frames_tensor, text_candidates[-1])
    # print(umt_score)
    

    
    
    