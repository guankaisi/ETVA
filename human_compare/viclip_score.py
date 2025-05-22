from third_party.viclip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from third_party.viclip.viclip import ViCLIP
import torch
import numpy as np
import cv2

clip_candidates = {'viclip':None, 'clip':None}

def get_clip(name='viclip'):
    global clip_candidates
    m = clip_candidates[name]
    if m is None:
        if name == 'viclip':
            tokenizer = _Tokenizer()
            vclip = ViCLIP(tokenizer)
            # m = vclip
            m = (vclip, tokenizer)
        else:
            raise Exception('the target clip model is not found.')
    
    return m

def get_text_feat_dict(texts, clip, tokenizer, text_feat_d={}):
    for t in texts:
        feat = clip.get_text_features(t, tokenizer, text_feat_d)
        text_feat_d[t] = feat
    return text_feat_d

def get_vid_feat(frames, clip):

    return clip.get_vid_features(frames)

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
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube
def CLIP_setup(name,device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    return clip, tokenizer

def ViCLIP_Score(frames_tensor, texts, clip, tokenizer, device=torch.device('cuda')):
    # clip, tokenizer = cli(name)
    # frames_tensor = frames2tensor(frames, device=device)
    vid_feat = get_vid_feat(frames_tensor, clip)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    
    clip_score =  vid_feat @ text_feats_tensor.T
    
    return clip_score.cpu().numpy()[0][0]

def retrieve_text(frames, texts, name='viclip', topk=5, device=torch.device('cuda')):
    clip, tokenizer = get_clip(name)
    clip = clip.to(device)
    frames_tensor = frames2tensor(frames, device=device)
    print(frames_tensor.shape)
    vid_feat = get_vid_feat(frames_tensor, clip)
    print(vid_feat.shape)
    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, clip, tokenizer, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = torch.cat(text_feats, 0)
    clip_score =  vid_feat @ text_feats_tensor.T
    probs, idxs = clip.get_predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.numpy()[0].tolist()]
    return ret_texts, probs.numpy()[0]
if __name__ == "__main__":
    video = cv2.VideoCapture('example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                    "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                    "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                    "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                    "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                    "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                    "A playful dog slides down a snowy hill, wagging its tail with delight.",
                    "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                    "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                    "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]

    texts, probs = retrieve_text(frames, text_candidates, name='viclip', topk=5)

    for t, p in zip(texts, probs):
        print(f'text: {t} ~ prob: {p:.4f}')