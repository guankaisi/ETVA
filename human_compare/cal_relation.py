import json
import scipy.stats

def CalulateRelation(json_list1, json_list2):
    # json_list1 and json_list2 are list of jsons, each json has the same keys, we calculate the kendall's tau and spearman correlation

    # Find common models
    common_models = set(json_list1[0].keys()) & set(json_list2[0].keys())
    kendall_tau_sum = 0
    spearman_corr_sum = 0
    for json1, json2 in zip(json_list1, json_list2):
        # Extract aligned scores
        scores1 = [json1[model] for model in common_models]
        scores2 = [json2[model] for model in common_models]
        # print(scores1,scores2)
        # Compute Kendall's Tau and Spearman's rank correlation
        kendall_tau, _ = scipy.stats.kendalltau(scores1, scores2)
        # print(type(kendall_tau))
        spearman_corr, _ = scipy.stats.spearmanr(scores1, scores2)
        if type(kendall_tau)== type(0.0):
            kendall_tau = 0.0
        if type(spearman_corr)== type(0.0):
            spearman_corr = 0.0
        # print(kendall_tau, spearman_corr)
        kendall_tau_sum += kendall_tau
        spearman_corr_sum += spearman_corr
    return kendall_tau_sum / len(json_list1), spearman_corr_sum / len(json_list1)


def load_result_list(file_path):
    with open(file_path, 'r') as f:
        json_result_list = [json.loads(line) for line in f.readlines()]
    types = ['existence','motion','material','spatial','number','shape','color','camera','physics','other']
    js_types = {}
    for type_ in types:
        js_types[type_] = []
    with open('result/prompts_category.jsonl', 'r') as f:
        category_list = [json.loads(line) for line in f.readlines()]
    for js1, js2 in zip(json_result_list, category_list):
        for type_ in types:
            if type_ in js2['category']:
                js_types[type_].append(js1)
    return js_types

if __name__ == "__main__":
    path = 'result/videoscore.json'
    human_path = 'result/human_result.json'
    compare_list = load_result_list(path)
    human_list = load_result_list(human_path)
    for key in compare_list.keys():
        print(f"Category: {key}")
        kendall_tau, spearman_corr = CalulateRelation(compare_list[key], human_list[key])
        print(f"Average Kendall's Tau: {kendall_tau:.4f}")
        print(f"Average Spearman's Rank Correlation: {spearman_corr:.4f}")
        print("================================")
    print(f"Overall")
    kendall_tau, spearman_corr = CalulateRelation([item for sublist in compare_list.values() for item in sublist], [item for sublist in human_list.values() for item in sublist])
    print(f"Average Kendall's Tau: {kendall_tau:.4f}")
    print(f"Average Spearman's Rank Correlation: {spearman_corr:.4f}")
    # print(compare_list.keys())
    # kendall_tau, spearman_corr = CalulateRelation(compare_list, human_list)
    # print(f"Average Kendall's Tau: {kendall_tau:.4f}")
    # print(f"Average Spearman's Rank Correlation: {spearman_corr:.4f}")