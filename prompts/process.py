import json
from tqdm import tqdm
# with open('/mnt/task_runtime/ETVA-T2valign/result/reason/reason_full.json', 'r') as f:
#     prompts1_list = [json.loads(line) for line in f.readlines()]

with open('/mnt/task_runtime/ETVA-T2valign/result/reason/reason_sample.json', 'r') as f:
    prompts1_list = [json.loads(line) for line in f.readlines()]


# with open('/mnt/task_runtime/ETVA-T2valign/result/reason/reason_full.json', 'w') as f:
#     for prompt_js1 in tqdm(prompts1_list):
#         prompt_js1['types'] = list(set(prompt_js1['types']))
#         f.write(json.dumps(prompt_js1)+'\n')

with open('/mnt/task_runtime/ETVA-T2valign/question_answer/model_outputs/sora_output_nocaption.jsonl', 'r') as f:
    prompts2_list = [json.loads(line) for line in f.readlines()]

with open('/mnt/task_runtime/ETVA-T2valign/result/reflect/sora_output.json', 'w') as f:
    for prompt_js1, prompt_js2 in tqdm(zip(prompts1_list, prompts2_list)):
        js_new = {
            'number': prompt_js1['number'],
            'prompt': prompt_js1['prompt'],
            'question_answer': prompt_js2['static_questions'] + prompt_js2['dynamic_questions'],
            'reasoning': prompt_js1['reasoning'],
            'types': prompt_js1['types']
        }
        f.write(json.dumps(js_new)+'\n')