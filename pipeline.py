import argparse
from tqdm import tqdm
import os
from etva.question_generation import Build_Scene_Graph, Question_Generation
from etva.question_answer import Reasoning_Stage, Reflection_Stage
from etva.utils import load_llm, load_mllm, load_prompts_from_json, load_prompts_from_txt

class Question_Generation_Pipeline():
    def __init__(self):
        super().__init__()

    def build_scene_graph(self, llm, prompt):
        return Build_Scene_Graph(prompt, llm)
    
    def question_generation(self, entity, attributes, relations, prompt, llm):
        return Question_Generation(entity, attributes, relations, prompt, llm)
    
    def question_generation_pipeline(self, llm, prompt):
        entity, attributes, relations = self.build_scene_graph(llm, prompt)
        questions = self.question_generation(entity, attributes, relations, prompt, llm)
        return questions

class Question_Answer_Pipeline():
    def __init__(self):
        super().__init__()

    def reasoning_stage(self, llm, prompt):
        reasoning = Reasoning_Stage(llm, prompt)
        return reasoning
    
    def reflection_stage(self, mllm, prompt, questions, reasoning, video_path, mllm_model):
        reflection_list = []
        for question in questions:  
            reflection = Reflection_Stage(mllm, question, reasoning, prompt, video_path, mllm_model)
            reflection_list.append(reflection)
        return reflection_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETVA-T2valign pipeline")
    parser.add_argument("--prompt_type", type=str, default='json', help="prompt type: json or txt")
    parser.add_argument("--prompt_path", type=str, default="./prompts/prompts_sample.json", help="path to the prompt file")
    parser.add_argument("--video_path", type=str, default="./video_folder/", help="path to the video file")
    parser.add_argument("--model_name", type=str, default="sora", help="video generationmodel name")
    parser.add_argument("--llm_path", type=str, default="./models/Qwen2.5-72B-Instruct", help="path to the llm model")
    parser.add_argument("--mllm_path", type=str, default="./models/Qwen2-VL-72B-Instruct", help="path to the mllm model")
    parser.add_argument("--question_path", type=str, default="./result/qa/qa_sample.json")
    parser.add_argument("--reason_path", type=str, default="./result/reason/reason_sample.json")
    # three tasks: qg for question generation, reason for reasoning, reflect for reflection
    parser.add_argument("--task", type=str, default="reflect", help="task: qg, reason, reflect")
    parser.add_argument("--output_path", type=str, default="./result/")
    args = parser.parse_args()

    if args.prompt_type == 'json':
        prompts_list = load_prompts_from_json(args.prompt_path)
    elif args.prompt_type == 'txt':
        prompts = load_prompts_from_txt(args.prompt_path)
    video_file_path = os.path.join(args.video_path, args.model_name)
    if args.task == "qg":
        llm = load_llm(args.llm_path)
        qg_pipeline = Question_Generation_Pipeline()
        for prompt_js in tqdm(prompts_list):
            prompt = prompt_js['prompt']
            questions = qg_pipeline.question_generation_pipeline(llm, prompt)
            prompt_js['questions'] = questions
            with open(os.path.join(args.output_path, 'qa', f"qa_sample.json"), "a+") as f:
                f.write(json.dumps(prompt_js)+'\n')

    if args.task == "reason":
        llm = load_llm(args.llm_path)
        qa_pipeline = Question_Answer_Pipeline()
        prompts_list = load_prompts_from_json(args.question_path)
        for prompt_js in tqdm(prompts_list):
            prompt = prompt_js['prompt']
            reasoning = qa_pipeline.reasoning_stage(llm, prompt)
            prompt_js['reasoning'] = reasoning
            with open(os.path.join(args.output_path, 'reason', "reason_sample.json"), "a+") as f:
                f.write(json.dumps(prompt_js)+'\n')
    
    if args.task == "reflect":
        mllm = load_mllm(args.mllm_path)
        qa_pipeline = Question_Answer_Pipeline()
        prompts_list = load_prompts_from_json(args.reason_path)
        for i, prompt_js in enumerate(tqdm(prompts_list)):
            prompt = prompt_js['prompt']
            questions = prompt_js['questions']
            reasoning = prompt_js['reasoning']
            video_path = os.path.join(video_file_path, f"{i}.mp4")
            reflection_list = qa_pipeline.reflection_stage(mllm, prompt, questions, reasoning, video_path, args.mllm_path)
            for question, reflection in zip(questions, reflection_list):
                q_and_a = []
                q_and_a.append({
                    "question": question,
                    "answer": reflection
                })
            prompt_js['question_answer'] = q_and_a
            with open(os.path.join(args.output_path, 'reflect', f"{args.model_name}_sample_output.json"), "a+") as f:
                f.write(json.dumps(prompt_js)+'\n')
            
