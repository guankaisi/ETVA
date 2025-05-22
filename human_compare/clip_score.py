import clip
import torch
import numpy as np
import cv2
v_mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break
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
class Clip_Score:   
    def __init__(self, path='/fs/fast/share/aimind_files/video_eval/models/ViT-B-16/ViT-B-16.pt'):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, _ = clip.load(path, self.device)
        self.clip_model.eval()  # 设置为评估模式
    
    def calculate_clip_score(self, frames_tensor, text):
        frames = frames_tensor.squeeze(0)
        batch_size = 4  # 调整批次大小以适应显存
        features = []
        with torch.no_grad():  # 禁用梯度
            for i in range(0, len(frames), batch_size):
                batch = frames[i:i + batch_size]
                batch_features = self.clip_model.encode_image(batch)
                features.append(batch_features)
            video_feature = torch.cat(features, dim=0).mean(dim=0)
            video_feature /= video_feature.norm()
            text_feature = self.clip_model.encode_text(clip.tokenize([text]).to(self.device))
            text_feature /= text_feature.norm(dim=-1, keepdim=True)
            return (video_feature @ text_feature.T).item()
            
if __name__ == "__main__":
    video = cv2.VideoCapture('example1.mp4')
    frames = [x for x in _frame_from_video(video)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_tensor = frames2tensor(frames, device=device)
    frames_tensor = frames_tensor.squeeze(0)
    path = '/fs/fast/share/aimind_files/video_eval/models/ViT-B-16/ViT-B-16.pt'
    clip_model, _ = clip.load(path, device)
    text = 'A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.'
    video_feature = clip_model.encode_image(frames_tensor).mean(dim=0)
    video_feature /= video_feature.norm()
    text_feature = clip_model.encode_text(clip.tokenize([text]).to(device))
    text_feature /= text_feature.norm(dim=-1, keepdim=True) 
    print(text_feature.shape)
    # print(clip_model)
    clip_score = video_feature @ text_feature.T
    print(clip_score.detach().cpu().numpy())