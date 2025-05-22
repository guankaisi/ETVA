from transformers import AutoProcessor
from etva.third_party.qwen_vl_utils import process_vision_info
from vllm import SamplingParams
import os
import json
from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import string

_PROMPT_TEMPLATE = string.Template("""
$preamble

$examples

$test_input_output
""".strip())

REASONING_STAGE_PROMPT = '''You are a helpful assistant tasked with analyzing video text prompts. Upon receiving a text prompt, your objective is to extract and identify implicit knowledge—such as common sense or relevant physical principles—that is not explicitly stated in the prompt but is necessary for generating an accurate and realistic video. You should present this knowledge as examples below.'''

REASONING_STAGE_EXAMPLE = """Example 1:
prompt: A cyclist rides through a bustling city street during rush hour, weaving between pedestrians and navigating around parked cars.
Implicit knowledge: 1.Kinetic Friction and Balance:
Video Representation: Show the cyclist maintaining balance by subtly shifting their body weight and adjusting the handlebars, especially when navigating tight spaces or making sharp turns.
2.Reaction Time and Decision Making:
Video Representation: Depict the cyclist making quick decisions, such as braking suddenly to avoid a pedestrian or swerving to bypass an obstacle, illustrating the importance of reaction time in traffic.
3.Traffic Flow and Human Behavior:
Video Representation: Illustrate the flow of traffic and pedestrian movement, showing how the cyclist anticipates and responds to the actions of others, such as stopping at a crosswalk or merging into traffic lanes.
4.Sound and Environmental Cues:
Video Representation: Incorporate ambient city sounds like honking horns, footsteps, and the cyclist’s breathing or gear shifting to convey the dynamic and potentially chaotic environment.

Example 2:
prompt: A snowboarder performs a series of tricks down a snow-covered mountain slope, showcasing agility and control.
Implicit knowledge: 1.Physics of Motion and Gravity:
Video Representation: Illustrate the snowboarder’s acceleration due to gravity and their ability to control speed and direction through body movements and board positioning.
2.Aerodynamics and Air Resistance:
Video Representation: Show the snowboarder adjusting their posture to minimize air resistance during high-speed sections or to gain height during jumps.
3.Impact Absorption and Safety Gear:
Video Representation: Highlight the snowboarder’s protective gear, such as helmets and padding, which absorb impact forces during landings and falls, ensuring safety.
4.Terrain Interaction and Snow Conditions:
Video Representation: Depict different snow textures and slopes, showing how the snowboarder adapts their technique to navigate through powder, ice, or moguls effectively.

Example 3:
prompt: A cup of water is slowly poured out in the space station, releasing the liquid into the surrounding area.
Implicit knowledge: 1.Microgravity Behavior of Liquids:
Video Representation: Show water separating from the cup into spherical droplets that float freely in the cabin instead of falling to the floor.
2.Surface Tension Effects
Video Representation: Highlight close-up shots of water droplets retaining their round shape as they detach from the stream.
3.Absence of Air Resistance
Video Representation: Depict droplets drifting gently across the cabin, moving in straight paths without slowing down quickly.

Example 4:
prompt: A beekeeper carefully harvests honey from a hive, ensuring minimal disturbance to the bee colony while collecting the golden liquid.
Implicit knowledge: 1.Bee Behavior and Colony Dynamics:
Video Representation: Show bees returning to the hive and interacting with each other, illustrating their organized behavior and the importance of maintaining the colony’s harmony during harvesting.
2.Thermal Properties of Honey:
Video Representation: Depict the beekeeper using a heated knife or uncapping tool to gently remove wax seals, demonstrating how controlled heat affects the viscosity and flow of honey.
3.Protective Equipment and Safety Measures:
Video Representation: Highlight the beekeeper’s protective gear, such as a suit, gloves, and veil, which safeguard against bee stings and ensure safe handling of the hive.
4.Extraction Process and Equipment Functionality:
Video Representation: Illustrate the use of an honey extractor, showing how centrifugal force is applied to extract honey without damaging the honeycombs, emphasizing the mechanical aspects of the process.

Example 5:
prompt: A firefighter battles a large blaze in a high-rise building, using a hose to spray water and rescuing trapped residents through smoke-filled corridors.
Implicit knowledge: 1.Fire Behavior and Heat Transfer:
Video Representation: Illustrate flames spreading vertically through stairwells and hallways, with hot air visibly rising and creating strong upward drafts.
2.Protective Gear Functionality:
Video Representation: Highlight the firefighter’s helmet, thermal-resistant suit, and breathing apparatus, showing how each component functions during the rescue.
3.Water Spray Dynamics:
Video Representation: Show close-ups of water being expelled from the hose in a powerful stream, effectively dousing flames and reducing ambient heat.
4.Evacuation Procedures and Team Coordination:
Video Representation: Depict firefighters working together to guide residents to safety, using hand signals or radios to communicate and coordinate their movements through smoke-filled areas.
"""

REASONING_TEST_INPUT_OUTPUT = """Example 6:
prompt: {prompt}
Implicit knowledge: """

GENERAL_ANSWER_PROMPT = '''You are a helpful video assistant. Now you are watching a video and given an answer to the question.'''

REFLECT_STAGE_PROMPT = '''You are a multimodal understanding assistant. You have access to the following:
1.Text (Input Description): A textual prompt that was used to generate the video.
2.Additional Common Sense Knowledge: Information that is not directly in the text but should be implicitly present in the video (e.g., logical or contextual details).
3.Video Clip: A generated video that you should analyze based on the text prompt and common sense. 

Your task is to answer a video-related question, but first, engage in thorough reflection by considering the following steps:
1.Video Understanding: Carefully interpret the video caption and any visible details, scenes, or information shown in the video, without referring to the text content.
2.Critical Reflection: Think through the implications of the video content in light of the text and common-sense knowledge. Does the video fully align with what was described, or are there gaps?
3.Conclusion: You should answer [YES] or [NO] special answer token to the question based on your analysis and provide a brief explanation to support your answer.
Here, the text prompt is: {prompt}
The common sense knowledge is: {common_sense}
And the question is: {question}
Please finish the Contextual Analysis, Video Understanding, Critical Reflection, and Conclusion stages with
[YES] or [NO] special answer token.'''

def process_reflection_video_inputs(
        video_model=None, 
        video_path=None, 
        prompt=None, 
        common_sense=None, 
        video_caption=None, 
        question=None):
    processor = AutoProcessor.from_pretrained(video_model)
    reflect_prompt = REFLECT_STAGE_PROMPT.format(prompt=prompt, common_sense=common_sense, video_caption=video_caption, question=question)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": reflect_prompt},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    _ , video_inputs = process_vision_info(messages)
    mm_data = {}
    mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    return llm_inputs

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    min_tokens=10,
    max_tokens=1024,
    stop_token_ids=[],
)

# def Reasoning_Stage(llm, prompt_batch):
#     reason_prompt_batch = [_PROMPT_TEMPLATE.substitute(
#         preamble=REASONING_STAGE_PROMPT,
#         examples=REASONING_STAGE_EXAMPLE,
#         test_input_output=REASONING_TEST_INPUT_OUTPUT.format(prompt=prompt)
#     ) for prompt in prompt_batch]

#     sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=150, repetition_penalty=1.2,stop=["example","Example","Input","To","Task","Entity","\n\n"])
#     outputs = llm.generate(reason_prompt_batch, sampling_params=sampling_params, use_tqdm=False)
#     reasoning_content = [output.outputs[0].text for output in outputs]
#     return reasoning_content

def Reasoning_Stage(llm, prompt):
    reasoning_prompt = _PROMPT_TEMPLATE.substitute(
        preamble=REASONING_STAGE_PROMPT,
        examples=REASONING_STAGE_EXAMPLE,
        test_input_output=REASONING_TEST_INPUT_OUTPUT.format(prompt=prompt)
    )
    reasoning = llm.generate(reasoning_prompt, sampling_params=sampling_params, use_tqdm=False)
    reasoning = reasoning[0].outputs[0].text
    return reasoning

def Reflection_Stage(mllm, question, reasoning, prompt, video_path, mllm_model):
    processor = AutoProcessor.from_pretrained(mllm_model)
    mllms_input = process_reflection_video_inputs(video_model=mllm_model, video_path=video_path, prompt=prompt, common_sense=reasoning, question=question)
    outputs = mllm.generate(mllms_input, sampling_params=sampling_params, use_tqdm=False)
    output = outputs[0].outputs[0].text
    return output