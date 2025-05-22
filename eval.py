import json
import argparse
from etva.utils import load_prompts_from_json, load_prompts_from_txt


# Calculate the Answer accuracy
def eval(eval_list):
    count_total = 0
    count_right = 0
    for eval_js in eval_list:
        q_a_list = eval_js['question_answer']
        for q_a_js in q_a_list:
            answer = q_a_js['answer']
            if '[YES]' in answer:
                count_right += 1
            count_total += 1
    
    print("ETVA Score: ", count_right / count_total)
            

        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETVA-T2valign pipeline")
    parser.add_argument("--prompt_type", type=str, default='json', help="prompt type: json or txt")
    parser.add_argument("--eval_path", type=str, default="./result/reflect/vidu_sample_output.json")
    args = parser.parse_args()

    if args.prompt_type == 'json':
        eval_list = load_prompts_from_json(args.eval_path)
    elif args.prompt_type == 'txt':
        eval_list = load_prompts_from_txt(args.eval_path)
    
    eval(eval_list)

# prompt1_list = load_prompts_from_json("/mnt/task_runtime/ETVA-T2valign/result/reflect/vidu_output_nocaption.jsonl")
# prompt2_list = load_prompts_from_json("/mnt/task_runtime/ETVA-T2valign/result/reflect/sora_sample_output.json")

# for prompt1_js, prompt2_js in zip(prompt1_list, prompt2_list):
#     q_a_list = []
#     q_a_list.extend(prompt1_js['static_questions'])
#     q_a_list.extend(prompt1_js['dynamic_questions'])
#     prompt2_js['question_answer'] = q_a_list

#     with open(f"/mnt/task_runtime/ETVA-T2valign/result/reflect/vidu_sample_output.json", "a+") as f:
#         f.write(json.dumps(prompt2_js))
#         f.write("\n")

    
    
    
    


