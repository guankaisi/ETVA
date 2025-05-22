import string
from pprint import pprint
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import os
import json
_PROMPT_TEMPLATE = string.Template("""
$preamble

$examples

$test_input_output
""".strip())

# We have four steps Entity Extraction, Entity Attribute Extraction, Relation Extraction, and Question Generation
_Entity_Extraction_PREAMBLE = """Task: given input prompts, extract the prompt background, camera and entities from the prompts.
Do not generate same entities multiple times. Do not generate entities that are not present in the prompts. Just output the entities and do not output other things.
output format: Background | "background"
Camera | "camera"
id | entity
""".strip()

_Entity_Attribute_Extraction_PREAMBLE = """Task: given input prompts and entity, extract the attributes of entities from the prompts.
Attributes are intrinsic characteristics of an entity and should not contain external entities that can be divided. Do not generate same attributes multiple times. Do not generate attributes that are not present in the prompts. Do not generate other entities as attributes. If no attribute is present, output "no mention".
output format: entity | attribute | value
"""

_Relation_Extraction_PREAMBLE = """Task: given input prompts and entity, extract the relations between entities from the prompts. Notice that the relations are at least between two entities and if there is only one entity, output "no mention".
Do not generate same relations multiple times. Do not generate relations that are not present in the prompts.
output format: id | entity1 | relation | entity2
"""

_Question_Generation_PREAMBLE = """Task: You are a helpful question generator for video. You are asked to generate questions based on the input video prompts and related entities, attributes and relations. Please ask questions as the format of examples. All the questions may can be answered by yes or no.
output format: question
"""

_Entity_Extraction_EXAMPLE = """Example 1:
input: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
output: 
Background | Harvest 
Camera | no mention
1 | bear
2 | cornfield
3 | stalk
4 | skateboarder
5 | urban skatepark

Example 2:
input: Pink motorcycle weaving through orange traffic cones, camera static.
output: 
Background | city road
Camera | static
1 | motorcycle
2 | traffic cone

Example 3:
input: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
output: 
Background | no mention
Camera | no mention
1 | man
2 | bicycle
3 | hoodie
4 | jeans
"""

_Entity_Extraction_TEST_INPUT_OUTPUT = """Example 4:
input: {prompt}
output: 
"""

_Entity_Attribute_EXAMPLE = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves. Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
all entities: bear, cornfield, stalk, skateboarder, urban skatepark
entity: bear
attributes: 
bear | number | one

Example 2:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all entitiess: man, bicycle, hoodie, jeans
entity: man
attiibutes: 
man | number | one
man | age | young
man | hair color | brown
man | hair style | messy
man | scar location | above left eye

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all entities: man, bicycle, hoodie, jeans
entity: hoodie
attributes: 
hoodie | color | blue 

Example 4:
prompt: Under the umbrella, a dancer with metallic skin twirling near a glowing tree
all_entities: umbrella, dancer, tree
entity: umbrella
attributes: 
umbrella | no mention

Example 5:  
prompt: A dancer with metallic skin twirls near a glowing tree.
all_entities: Dancer, tree
entity: Dancer
attributes: 
Dancer | number | one
Dancer | skin | metallic
"""

Entity_Attribute_TEST_INPUT_OUTPUT = """Example 6:
prompt: {prompt}
all_entities: {all_entities}
entity: {entity}
attributes: 
"""

_Relation_Extraction_EXAMPLE = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
all_entities: bear, cornfield, stalk, skateboarder, urban skatepark
relations: 1 | bear | rampages | cornfield
2 | stalk | collapsing | cornfield
3 | skateboarder | tearing | urban skatepark 

Example 2:
prompt: Pink motorcycle weaving through orange traffic cones, camera static.
all_entities: motorcycle, traffic cone
relations: 1 | motorcycle | weaving | traffic cone

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
all_entities: man, bicycle, hoodie, jeans
relations: 1 | man | riding | bicycle
2 | man | wearing | hoodie
3 | man | wearing | jeans

Example 4:
prompt: a girl is walking forward, /camera push in.
all_entities: girl
relations: no mention
"""

Relation_Extraction_TEST_INPUT_OUTPUT = """Example 5:
prompt: {prompt}
all_entities: {all_entities}
relations: """

Question_Generation_Example = """Example 1:
prompt: During harvest, a bear rampages through a cornfield, stalks collapsing in waves.Film a group of skateboarders tearing through an urban skatepark, performing flips, grinds, and tricks with lightning-fast agility.
question_type : entity (entity, attribute, relation)
content: Background | Harvest 
question: Is the video background in the scene of Harvest?

Example 2: 
prompt: Pink motorcycle weaving through orange traffic cones, camera static.
question_type : entity (entity, attribute, relation)
content: 1 | motorcycle
question: Does the video show a motorcycle?

Example 3:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
question_type : attribute (entity, attribute, relation)    
content: man | hair color | brown
question: Is the hair color of the man brown?

Example 4:
prompt: A young man is riding a bicycle. He is wearing a blue hoodie and black jeans. His hair is brown and styled messily. He has a small scar above his left eye.
question_type : relation (entity, attribute, relation)
content: 3 | man | riding | bicycle
question: Is the man riding a bicycle?"""


Question_Generation_TEST_INPUT_OUTPUT = """Example 5:
prompt: {prompt}
question_type: {question_type} (entity, attribute, relation)
content: {content}
question: """

_ENTITY_EXTRACTION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Entity_Extraction_PREAMBLE,
    examples=_Entity_Extraction_EXAMPLE,
    test_input_output=_Entity_Extraction_TEST_INPUT_OUTPUT
)

_ENTITY_ATTRIBUTE_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Entity_Attribute_Extraction_PREAMBLE,
    examples=_Entity_Attribute_EXAMPLE,
    test_input_output=Entity_Attribute_TEST_INPUT_OUTPUT
)

_RELATION_EXTRACTION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Relation_Extraction_PREAMBLE,
    examples=_Relation_Extraction_EXAMPLE,
    test_input_output=Relation_Extraction_TEST_INPUT_OUTPUT
)

_QUESTION_GENERATION_PROMPT = _PROMPT_TEMPLATE.substitute(
    preamble=_Question_Generation_PREAMBLE,
    examples=Question_Generation_Example,
    test_input_output=Question_Generation_TEST_INPUT_OUTPUT
)

def Build_Scene_Graph(prompt,llm):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=150, repetition_penalty=1.2,stop=["example","Example","Input","prompt","To","Prompt","Task","Entity","\n\n"])
    # Task 1: Entity Extraction
    task1_prompt = _ENTITY_EXTRACTION_PROMPT.format(prompt=prompt)
    task1_outputs = llm.generate([task1_prompt], sampling_params=sampling_params, use_tqdm=False)
    task1_outputs = [output.outputs[0].text for output in task1_outputs]
    # Task 2: Entity Attribute Extraction
    all_entity_list = [item for item in task1_outputs[0].split("\n") if "|" in item][2:]
    entity_list = [item for item in task1_outputs[0].split("\n") if "|" in item]
    all_entities = ", ".join([item.split("|")[1].strip() for item in all_entity_list])
    task2_prompts = []
    for entity in entity_list:
        entity = entity.split("|")[1].strip()
        task2_prompts.append(_ENTITY_ATTRIBUTE_PROMPT.format(prompt=prompt, all_entities=all_entities,entity=entity))
        
    task2_outputs = llm.generate(task2_prompts, sampling_params=sampling_params, use_tqdm=False)
    task2_outputs = [output.outputs[0].text for output in task2_outputs]
    # Task 3: Relation Extraction
    task3_prompt = _RELATION_EXTRACTION_PROMPT.format(prompt=prompt, all_entities=all_entities)
    task3_outputs = llm.generate([task3_prompt], sampling_params=sampling_params, use_tqdm=False)
    task3_outputs = [output.outputs[0].text for output in task3_outputs]
    entity = [entity for entity in task1_outputs[0].split("\n") if "|" in entity]
    attributes = []
    for output in task2_outputs:
        attributes += [attribute for attribute in output.split("\n") if "|" in attribute]

    relations = [relation for relation in task3_outputs[0].split("\n") if "|" in relation]
    return entity, attributes, relations

def Question_Generation(entity, attributes, relations, prompt, llm):
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=150, repetition_penalty=1.2,stop=["example","Example","Input","To","Task","Entity","\n\n"])
    # Question Generation entity
    entity_prompts = []
    for item in entity:
        if "no mention" in item.lower():
            continue
        entity_prompts.append(_QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="entity", content=item))
    attribute_prompts = []
    for item in attributes:
        if "no mention" in item.lower():
            continue
        attribute_prompts.append(_QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="attribute", content=item))
    relation_prompts = []
    for item in relations:
        if "no mention" in item.lower():
            continue
        relation_prompts.append(_QUESTION_GENERATION_PROMPT.format(prompt=prompt, question_type="relation", content=item))
    entity_outputs = llm.generate(entity_prompts, sampling_params=sampling_params, use_tqdm=False)
    attribute_outputs = llm.generate(attribute_prompts, sampling_params=sampling_params, use_tqdm=False)
    relation_outputs = llm.generate(relation_prompts, sampling_params=sampling_params, use_tqdm=False)

    entity_questions = [output.outputs[0].text for output in entity_outputs]
    attribute_questions = [output.outputs[0].text for output in attribute_outputs]
    relation_questions = [output.outputs[0].text for output in relation_outputs]
    questions = entity_questions + attribute_questions + relation_questions
    return questions


