"""
pip install qwen_vl_utils mantis-vl
"""
import torch
from mantis.models.qwen2_vl import Qwen2VLForSequenceClassification
from transformers import Qwen2VLProcessor
from qwen_vl_utils import process_vision_info
import gc
import numpy as np
ROUND_DIGIT=3
REGRESSION_QUERY_PROMPT = """
Suppose you are an expert in judging and evaluating the quality of AI-generated videos,
please watch the following frames of a given video and see the text prompt for generating the video,
then give scores from 5 different dimensions:
(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color
(2) temporal consistency, both the consistency of objects or humans and the smoothness of motion or movements
(3) dynamic degree, the degree of dynamic changes
(4) text-to-video alignment, the alignment between the text prompt and the video content
(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge

for each dimension, output a float number from 1.0 to 4.0,
the higher the number is, the better the video performs in that sub-score, 
the lowest 1.0 means Bad, the highest 4.0 means Perfect/Real (the video is like a real video)
Here is an output example:
visual quality: 3.2
temporal consistency: 2.7
dynamic degree: 4.0
text-to-video alignment: 2.3
factual consistency: 1.8

For this video, the text prompt is "{text_prompt}",
all the frames of video are as follows:
"""    
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
def load_video_score_model(model_name):
    model = Qwen2VLForSequenceClassification.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()  # 添加eval模式
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    return model, processor

def VideoScore(model, processor, video_path, video_prompt):
    # 添加内存清理装饰器
    @torch.no_grad()  # 确保不保存计算图
    def _score():
        response = ""
        label_names = ["visual quality", "temporal consistency", "dynamic degree", 
                      "text-to-video alignment", "factual consistency"]
        labels = ['LABEL_0','LABEL_1','LABEL_2','LABEL_3','LABEL_4']
        for i in range(len(label_names)):
            response += f"The score for {label_names[i]} is {labels[i]}. "
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "min_pixels": 224 * 224,
                        "max_pixels": 224 * 224,
                        "num_frames": 5  # 显式限制帧数
                    },
                    {"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response},
                ],
            }
        ]

        # 处理输入时添加清理机制
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        fnum = 8
        step = len(video_inputs[0]) // fnum
        if step == 0:
            step = 1
        video_inputs = [video_inputs[0][::step][:fnum]]
        inputs = processor(
            text=[text],
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        # print(inputs['pixel_values_videos'].shape)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
                
            aspect_scores = [round(logits[0, i].item(), ROUND_DIGIT) for i in range(logits.shape[-1])]
            return aspect_scores[-2]  # 返回text-to-video alignment分数
            
       


    return _score()


if __name__ == "__main__":
    model_name="/fs/fast/share/aimind_files/video_eval/models/VideoScore-Qwen2-VL"
    video_path="/fs/fast/share/aimind_files/human_eval/static/video_folder/sora/45.mp4"
    video_prompt="A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area"


    model = Qwen2VLForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map="auto"
    )


    # default processer
    processor = Qwen2VLProcessor.from_pretrained(model_name)

    # Messages containing a images list as a video and a text query
    response = ""
    label_names = ["visual quality", "temporal consistency", "dynamic degree", "text-to-video alignment", "factual consistency"]
    labels = ['LABEL_0','LABEL_1','LABEL_2','LABEL_3','LABEL_4']
    for i in range(len(label_names)):
        response += f"The score for {label_names[i]} is {labels[i]}. "
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                    "num_frames": 8,
                },
                {"type": "text", "text": REGRESSION_QUERY_PROMPT.format(text_prompt=video_prompt)},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": response},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    video_inputs = [video_inputs[0][:10]]
    print(video_inputs[0].shape)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # print(inputs.shape)
    # inputs['pixel_values_videos'] = inputs['pixel_values_videos'][:10200]
    # print(inputs['pixel_values_videos'].shape)
    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    num_aspects = logits.shape[-1]

    aspect_scores = []
    for i in range(num_aspects):
        aspect_scores.append(round(logits[0, i].item(),ROUND_DIGIT))
    print(aspect_scores)

    """
    model output on visual quality, temporal consistency, dynamic degree,
    text-to-video alignment, factual consistency, respectively
    VideoScore: 
    [2.297, 2.469, 2.906, 2.766, 2.516]

    VideoScore-Qwen2-VL:
    [2.297, 2.531, 2.766, 2.312, 2.547]
    """
