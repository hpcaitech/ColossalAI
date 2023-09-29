# GPT Evaluation
## Table of Contents
- [Overview](#overview)
- [GPT Evaluation](#gpt-evaluation)
  - [Evaluation Category](#evaluation-category)
  - [Evaluation Category Examples](#evaluation-category-examples)
  - [Evaluation Metrics](#evaluation-metrics)
- [Evaluation Process](#evaluation-process)
  - [Data Format](#data-format)
  - [Prompt](#prompt)
    - [Battle Prompt](#battle-prompt)
    - [Evaluation Prompt](#evaluation-prompt)
  - [Evaluation](#evaluation)
    - [Configuration](#configuration)
    - [Evaluate](#evaluate)
- [FAQ](#faq)
- [Citations](#citations)


## Overview

In this directory, we introduce how you can evaluate your model using GPTs. It is now available for evaluation of both Chinese and English capability and we provide the following functions:

* Compare the performance of two different models (battle).
* Rate the model according to pre-defined metrics using prompting design.
* Rate the model according to pre-defined metrics with additional reference answer using prompting design.

## GPT Evaluation

### Evaluation Category

Our evaluation pipeline can examine the model's capability using different categories of questions. The following table includes some example categories. You can add your own questions.

| Evaluation Category | Description                                                  |
| :-----------------: | :----------------------------------------------------------- |
|    Brainstorming    | Models are asked to generate a range of creative and diverse ideas according to the question. The capability of creativity is required. |
|        Chat         | Models are asked to continue a multi-round dialogue given the roles involved. The capability of understanding, memorizing previous rounds of the dialogue and answering according to the persona provided is required. |
|     Generation      | Models are asked to generate an email, letter, article, etc. The capability of generating texts in a high quality and human-written way is required. |
|       Open QA       | Models are asked to answer an open QA question(without context provided). The capability of answering questions with the models' own knowledge base is required. |
|       Roleplay      | Models are asked to play the role provided. The capability of engaging in the scenario and effectively interacting with the user is required. |


### Evaluation Category Examples
To better understand each evaluation category, here are some example questions provided. Example questions are in the `configs/gpt_evaluation/data` folder.


| Evaluation Category | Chinese Example                                              | English Example                                              |
| :-----------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|    Brainstorming    | 列举一些可以促进头发生长的食物。                             | How do you properly chop an onion without crying?            |
|        Chat         | 基于以下角色信息完成一段对话。小张是一名新手爱好者，对养鸡有浓厚的兴趣。老李是一名有丰富经验的养鸡大师。<br/>小张：您好，老李，我最近开始对养鸡感兴趣了，想请教您一些问题。 <br/>老李：你好，小张，我很乐意帮助你。你想问些什么？ <br/>小张：我想知道如何确定鸡的品种和性别？ <br/>老李：确切的品种可以通过鸡的外貌特征来确定，而性别一般是通过鸡卵的大小和形状来判断。还有什么问题吗？<br/> 小张：<br/> | Complete a dialogue based on the following character information. Alex: A novice writer who is struggling to find inspiration and develop his writing skills. Emma: A successful author with many published works, providing guidance and advice to Alex.<br/>Alex: Hi Emma, I have been writing for a while now but can't seem to make any progress. Can you give me any advice? <br/>Emma: Hi Alex, sure. What kind of writing are you doing?<br/>Alex: I'm trying to write a novel, but I just can't seem to find any inspiration.<br/>Emma: <br/> |
|     Generation      | 请为一家咖啡店编写一篇简短的广告语，吸引更多的顾客。         | Write a set of guidelines for first-time pet owners on how to properly care for a new puppy. |
|       Open QA       | 解释什么是RNA病毒和DNA病毒。                                 | Explain the process of osmosis in biological systems.        |
|      Roleplay       | 我要你把我写的句子翻译成表情符号。我会写句子，你会用表情符号表达它。我只是想让你用表情符号来表达它。除了表情符号，我不希望你回复任何内容。当我需要用中文告诉你一些事情时，我会用 {} 这样的大括号括起来。我的第一句话是“{我的职业是消防员。}” | I want you to act as a rapper. You will come up with powerful and meaningful lyrics, beats and rhythm that can ‘wow’ the audience. Your lyrics should have an intriguing meaning and message which people can relate too. When it comes to choosing your beat, make sure it is catchy yet relevant to your words, so that when combined they make an explosion of sound everytime! My first request is "I need a rap song about finding strength within yourself." |

### Evaluation Metrics

GPT evaluation uses GPT models to evaluate the prediction of different models and different pre-defined evaluation metrics are applied to different categories. The following table shows the 10 pre-defined evaluation metrics both in Chinese and English:

|   Evaluation Metric   | Prompt Words                                                 | CoT(Chain-of-Thought)                                        |
| :-------------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
| 语言组织<br/>(Language organization) | 语言组织(1-5)：答案语言是否流畅、连贯，使用正确的语法，具有一定逻辑性，使用恰当的连接词、过渡词等等。</br></br>Language organization (1-5): whether the answer language is fluent and coherent, uses correct grammar, has a certain logic, uses appropriate connecting words, transition words, etc. | 1. 阅读答案，并检查是否有语法错误、用词不当或其他显著的错误。<br/> 2. 检查答案是否具有逻辑性，能够按照合理的顺序传达信息并且能够自圆其说<br/> 3. 确定答案是否与问题或主题相关，并且能够传达清晰的信息。<br/> 4. 检查答案是否连贯，是否使用适当的转换和过渡来保持句子和段落之间的连贯性。<br/> 5. 检查答案是否具有明确的结构和组织方式，使得读者可以轻松理解信息的层次和结构。<br/> 6. 根据以上因素综合评估答案的语言组织，并给出一个1到5的分数，其中5表示语言组织非常好，而1表示语言组织非常差。</br></br>1. Read the answers and check for grammatical errors, poor word choice, or other significant mistakes.<br>2. Check that the answer is logical, conveys the information in a logical order, and is self-explanatory.<br>3. Determine if the answer is relevant to the question or topic and conveys a clear message.<br>4. Check that the answer is coherent and that appropriate transitions and switches are used to maintain coherence between sentences and paragraphs.<br>5. Check that the answer is clearly structured and organized in such a way that the reader can easily understand the hierarchy and structure of the information.<br>6. Evaluate the linguistic organization of the answer based on a combination of the above factors and give a score of 1 to 5, where 5 indicates very good linguistic organization and 1 indicates very poor linguistic organization. |
|       切题<br/>(Relevance)       | 切题(1-5)：答案内容是否切题，不答非所问，并且严格遵照题目要求。</br></br>Relevance (1-5): whether the content of the answer is relevant to the topic, does not answer the wrong question, and strictly follows the requirements of the topic. | 1. 阅读题目，确定题目所问的问题是什么，以及需要回答哪些方面的问题。<br/> 2. 阅读答案，确认答案是否直接回答了题目所问的问题。<br/> 3. 检查答案是否严格遵照了题目的要求，包括答题方式、答题长度、答题格式等等。<br/> 4. 根据以上因素综合评估答案的切题程度，并给出一个1到5的分数，其中5表示答案非常切题，而1表示答案完全没有切题。</br></br>1. Read the question to determine what the question asks and what aspects of the question need to be answered.<br>2. Read the answers to make sure that they directly answer the question asked.<br>3. Check that the answer follows the requirements of the question, including the way it is answered, the length of the answer, the format of the answer, etc.<br>4. Evaluate how relevant the answer is based on the above factors and give a score of 1 to 5, where 5 means the answer is very relevant and 1 means the answer is not relevant at all. |
|      创意性<br/>(Creativity)       | 创意性(1-5)：某些头脑风暴问题可能需要答案具有创意，提出新的思路。</br></br>Creativity (1-5): Some brainstorming questions may require answers that are creative and suggest new ideas. | 1. 仔细阅读所提供的头脑风暴问题，确保你理解问题的要点和背景。<br/> 2. 根据你的知识和经验，判断所提供的答案是否可行。如果答案不可行，则创意性评分可能会受到影响。<br/> 3. 考虑答案中是否包含新颖的想法或独特的思路。答案可能与已知的解决方案有所重叠，但仍然可以被认为是有创意的，只要它提供了新的角度或方法来解决问题。<br/> 4. 根据答案的创意性，给出一个1到5的评分。如果答案缺乏创意，则应给出一个较低的评分。如果答案具有创意并提供了新的思路，应给出一个较高的评分。</br></br>1. Read the provided brainstorming questions carefully to make sure you understand the gist and context of the questions.<br>2. Based on your knowledge and experience, determine if the answers provided are feasible. If the answer is not feasible, the creativity score may be affected.<br>3. Consider whether the answer contains novel ideas or unique thoughts. An answer may overlap with a known solution and still be considered creative, as long as it offers a new perspective or approach to the problem.<br>4. Give a score of 1 to 5 depending on the creativity of the answer. If the answer lacks creativity, a lower score should be given. If the answer is creative and provides a new idea, a higher score should be given. |
|     实用性<br/>(Practicality)      | 实用性(1-5)：某些头脑风暴问题可能需要答案提出实用的建议或解决方法。</br></br>Practicality (1-5): Some brainstorming questions may require answers to suggest practical suggestions or solutions. | 1. 仔细阅读所提供的头脑风暴问题，确保你理解问题的要点和背景。<br/> 2. 根据你的知识和经验，判断所提供的答案是否可行。如果答案不可行，则实用性评分可能会受到影响。<br/> 3. 考虑答案中提出的建议或解决方法是否实用并可行。答案可能看起来很好，但如果无法实现或应用，则实用性评分可能会受到影响。<br/> 4. 根据答案的实用性，给出一个1到5的评分。如果答案缺乏实用性，则应给出一个较低的评分。如果答案提出了实用的建议或解决方法，并且可以很好地解决问题，则应给出一个较高的评分。</br></br>1. Read the provided brainstorming questions carefully to make sure you understand the gist and context of the questions.<br>2. Based on your knowledge and experience, determine if the answers provided are feasible. If the answer is not feasible, the practicality score may be affected.<br>3. Consider whether the suggestions or solutions presented in the answer are practical and workable. The answer may look good, but if it cannot be implemented or applied, the practicality score may be affected.<br>4. Give a score of 1 to 5 depending on the practicality of the answer. If the answer lacks practicality, a lower score should be given. If the answer makes a practical suggestion or solution and solves the problem well, a higher score should be given. |
|      正确性<br/>(Correctness)      | 正确性(1-5)：正确性(1-5)：答案是否正确。</br></br> Correctness (1-5): whether the answer is correct or not. | 1. 仔细阅读题目，尝试自己回答该问题。<br/>2. 检查答案的准确性。您可以使用已知的事实或研究来验证答案是否正确。如果答案是正确的，则可以将正确性得分为5分。如果答案是部分正确的，则可以给予适当的得分，例如2分、3分或4分。如果答案完全不正确，则只得1分。<br/><br/>1. Read the question carefully and try to answer the question yourself. <br/>2. Check the correctness of the answer. You can use known facts or research to verify that the answer is correct. If the answer is correct, you can give a score of 5 for correctness. If the answer is partially correct, an appropriate score, such as 2, 3, or 4, may be given. If the answer is completely incorrect, only 1 point is awarded. |
|      自然<br/>(Naturalness)      | 自然(1-5)：答案是否自然，并且符合问题给定的身份。</br></br>Naturalness (1-5): whether the answer is natural and fits the identity given by the question. | 1. 阅读题目，确定题目提供的身份信息。<br/> 2. 检查答案内容是否符合题目给定的身份。<br/> 3. 根据以上因素，对该回答的自然性进行打分，分数从1到5，其中1表示不自然，5表示非常自然，并符合问题给定的身份。</br></br>1. Read the question and determine the identity information provided in the question.<br>2. Check whether the content of the answer matches the identity given in the question.<br>3. Based on the above factors, score the naturalness of the response on a scale from 1 to 5, where 1 means unnatural and 5 means very natural and in accordance with the identity given in the question. |
|     参与感<br/>(Engagingness)      | 参与感(1-5)：答案是否对前面的对话内容做出了恰当的反应，是否理解对话的语境和背景。</br></br>Engagingness (1-5): whether the answer responds appropriately to the content of the preceding conversation and whether it understands the context and background of the conversation. | 1. 阅读题目，确定对话的语境和背景。<br/> 2. 检查答案是否充分理解对话的语境和背景，能否自然地融入到对话中而不显得突兀。<br/> 3. 根据以上因素，对该回答的参与感进行打分，分数从1到5，其中1表示没有参与感，5表示非常有参与感，并且恰当地理解了对话的语境和背景。</br></br>1. Read the questions to determine the context and background of the dialogue.<br>2. Check that the answer fully understands the context and background of the conversation and that it fits naturally into the conversation without seeming abrupt.<br>3. Based on the above factors, rate the response's engagement on a scale from 1 to 5, where 1 means not engaged and 5 means very engaged and appropriately understands the context and background of the conversation. |
|    合理性<br/>(Reasonableness)     | 合理性(1-5)：答案是否能够与前面的对话内容形成逻辑上的衔接，是否符合常理，能否在这个上下文中合理存在。</br></br>Reasonableness (1-5): Whether the answer can form a logical connection with the content of the previous dialogue, whether it is consistent with common sense, and whether it can reasonably exist in this context. | 1. 阅读题目，确定对话的主题以及问题期望的回答方向。<br/> 2. 判断答案是否能够与前面的对话内容形成逻辑上的衔接，是否符合常理，能否在这个上下文中合理存在。<br/> 3. 根据以上因素，对该回答的合理性进行打分，分数从1到5，其中1表示不合理，5表示非常合理，并且能够与前面的对话内容形成逻辑上的衔接，并符合常理。</br></br>1. Read the question and determine the topic of the conversation and the direction the question expects the answer to go.<br>2. Determine whether the answer can be logically connected to the preceding conversation, whether it makes common sense, and whether it can reasonably exist in this context.<br>3. Based on the above factors, rate the reasonableness of the answer on a scale from 1 to 5, where 1 means unreasonable and 5 means very reasonable and able to form a logical connection with the preceding dialogue content and consistent with common sense. |
|       多样性<br/>(Diversity)       | 多样性(1-5)：答案使用语言是否优美，具有有一定的创造性和想象力。然而，回答也应该保持合理和适度，不要过于夸张或离题。</br></br>Diversity (1-5): Whether the answers use beautiful language and have some creativity and imagination. However, answers should also be kept reasonable and moderate, not overly exaggerated or off-topic. | 1. 仔细阅读整个回答，确保完全理解回答所表达的内容和主题。<br/> 2. 在阅读回答的同时，注意语言的质量，例如措辞是否正确，语言是否生动等。<br/> 3. 检查回答的创造性和想象力，看看回答是否能够吸引人阅读下去。<br/> 4. 检查回答的合理性和适度，看看回答是否夸张或离题。5. 将多样性的评分打分在1到5之间，5分表示回答的质量很好，能够吸引人阅读，1分表示回答的内容生硬或者有离题的问题。</br></br>1. Read the entire response carefully to ensure that you fully understand the content and theme expressed in the response.<br>2. While reading the response, pay attention to the quality of the language, such as whether the wording is correct and the language is vivid.<br>3. Check the creativity and imagination of the response to see if the response is engaging to read on.<br>4. Check the reasonableness and appropriateness of the responses to see if the responses are exaggerated or off-topic.<br>5. Rate the diversity on a scale of 1 to 5, with a 5 indicating a good quality response that is engaging to read and a 1 indicating a raw response or a question that is off-topic. |
|       保真度<br/>(Fidelity)        | 保真度(1-5)：答案是否能够严格遵守角色的设定回答给定的请求。</br></br>Fidelity (1-5): whether the answer is able to answer the given request in strict compliance with the role setting. | 1. 仔细阅读问题，了解角色在问题中的设定和表现，包括职业、背景、观点、性格等方面。<br/> 阅读题目的请求，确认回答请求时需要注意的细节。<br/> 3. 对比提供的回答与该角色的设定，评估回答是否能够严格遵守角色的设定。<br/> 4. 结合以上评估结果给出保真度的评分，范围从1到5分，其中1分表示回答与角色设定完全不符，5分表示回答完全符合角色设定且满足给定请求。</br></br>1. Read the question carefully to understand how the character is set up and represented in the question, including aspects such as occupation, background, point of view, and personality.<br>2. Read the question's request and confirm the details that need to be taken into account when answering the request.<br>3. Compare the provided answer with the setting of the role and assess whether the answer can strictly adhere to the setting of the role.<br>4. Combine the results of the above assessment to give a fidelity score ranging from 1 to 5, where a score of 1 means that the response does not match the persona at all, and a score of 5 means that the response fully complies with the persona and satisfies the given request. |

GPT models evaluate the quality of model predictions based on the given prompt words and gives a score between 1-5.

> **NOTE 1:**  You can find all the prompt words and CoT(Chain-of-Thought) in `configs/gpt_evaluation/prompt/evaluation_prompt`.

> **NOTE 2:** To add customized metrics, you can refer to [FAQ](#faq).

## Evaluation Process

### Data Format

A JSON file contains one list. Each element in the list is a target answer / prediction record for one instruction / question.
An element should have the following fields:

* `category` (str, compulsory): The category of the instruction / question.
* `instruction` (str, compulsory): The instruction / question for the LLM.
* `input` (str, optional): The additional context of the instruction / question.
* `output` (str, optional): The model output of the instruction, models will fill in this field during inference time.
* `target` (str, optional): The target answer for the instruction.
* `id` (int, compulsory): The ID of the instruction / question.

Example:

```json
[
    {
        "category": "brainstorming",
        "instruction": "请问如何制作一份美味的西红柿炒鸡蛋？",
        "input": "",
        "output": "",
        "target": "",
        "id": 1
    },
    {
        "category": "chat",
        "instruction": "基于以下角色信息完成一段对话。小张是一名新手爱好者，对养鸡有浓厚的兴趣。老李是一名有丰富经验的养鸡大师。",
        "input": "小张：您好，老李，我最近开始对养鸡感兴趣了，想请教您一些问题。 老李：你好，小张，我很乐意帮助你。你想问些什么？ 小张：我想知道如何确定鸡的品种和性别？ 老李：确切的品种可以通过鸡的外貌特征来确定，而性别一般是通过鸡卵的大小和形状来判断。还有什么问题吗？ 小张：",
        "output": "",
        "target": "",
        "id": 2
    }
]
```

### Prompt

#### Battle Prompt

The following is the Chinese battle prompt. In the battle prompt, the question and answers from two different models are fed into the prompt template. You can find example battle prompt files for Chinese and English in `configs/gpt_evaluation/prompt/battle_prompt`.

```json
{
  "id": 1,
  "system_prompt": "你是一个检查回答质量的好助手。",
  "prompt_template": "[问题]\n{question}\n\n[1号AI助手的答案]\n{answer_1}\n\n[1号AI助手答案终止]\n\n[2号AI助手的答  案]\n{answer_2}\n\n[2号AI助手答案终止]\n\n[要求]\n{prompt}\n\n",
  "prompt": "我们需要你评价这两个AI助手回答的性能。\n请对他们的回答的有用性、相关性、准确性、详细程度进行评分。每个AI助手都会得到一个1到10分的总分，分数越高表示整体表现越好。\n请首先输出一行，该行只包含两个数值，分别表示1号和2号AI助手的分数。这两个分数之间要有一个空格。在随后的一行中，请对你的评价作出全面的解释，避免任何潜在的偏见，并确保AI助手回答的顺序不会影响您的判断。"
}
```

#### Evaluation Prompt

The following is an example of a Chinese GPT evaluation prompt. In an evaluation prompt, you should define your metrics in `metrics` and provide CoT(Chain-of-Thought) in `CoT`.  You can find example evaluation prompt files for Chinese and English in `configs/gpt_evaluation/prompt/evaluation_prompt`.

```json
{
  "brainstorming": {
    "id": 1,
    "category": "brainstorming",
    "metrics": {
      "language organization": "语言组织(1-5)：答案语言是否流畅、连贯，使用正确的语法，具有一定逻辑性，使用恰当的连接词、过渡词等等。"
    },
    "CoT": {
      "language organization": "1. 阅读答案，并检查是否有语法错误、用词不当或其他显著的错误。\n2. 检查答案是否具有逻辑性，能够按照合理的顺序传达信息并且能够自圆其说。\n3. 确定答案是否与问题或主题相关，并且能够传达清晰的信息。\n4. 检查答案是否连贯，是否使用适当的转换和过渡来保持句子和段落之间的连贯性。\n5. 检查答案是否具有明确的结构和组织方式，使得读者可以轻松理解信息的层次和结构。\n6. 根据以上因素综合评估答案的语言组织，并给出一个1到5的分数，其中5表示语言组织非常好，而1表示语言组织非常差。\n\n语言组织："
    },
    "prompt": "你是一个好助手。请你为下面“头脑风暴”问题的答案打分。\n\n问题如下：\n\n{question}\n\n答案如下：\n\n{answer}\n\n评分的指标如下：\n\n{metric}\n\n请你遵照以下的评分步骤：\n\n{steps}"
  }
}
```

`"metrics"`: the metrics that can be used in GPT evaluation. This field determines which metrics can be added to your config file.

`"CoT"`: evaluation steps you prompt to GPT models for each metric defined in `"metrics"`.

### Evaluation

#### Configuration

The following is an example of a Chinese config file. The configuration file can control how the pipeline evaluates the model. You need to specify GPT evaluation metrics in key `GPT`. You can find an example English config file in `configs/gpt_evaluation/config/config_en.json`.

```json
{
    "language": "cn",
    "category": {
        "brainstorming": {
            "GPT": [
                "language organization",
                "relevance",
                "creativity",
                "practicality",
                "reasonableness"
            ]
        }
    }
}
```

`"language"`: the language used to evaluate the model capability. We only support Chinese `"cn"` for now.

`"category"`: the category/categories needed to evaluate the model capability.

`"GPT"`: the metrics you want to use for GPT evaluation.


#### Evaluate

After setting the configuration file, you can evaluate the model using `examples/gpt_evaluation/eval.py`. If you want to make comparisons between answers of two different models, you should specify two answer files in the argument `answer_file_list` and two model names in the argument `model_name_list`. If you want to evaluate one answer file, the length of both `answer_file_list` and `model_name_list` should be 1 and the program will perform evaluation using automatic metrics and GPT models.

An example script is provided as follows:

```shell
python eval.py \
    --config_file "path to the config file" \
    --battle_prompt_file "path to the prompt file for battle" \
    --gpt_evaluation_prompt_file "path to the prompt file for gpt evaluation" \
    --target_file "path to the target answer file" \
    --answer_file_list "path to the answer files of at most 2 models" \
    --model_name_list "the names of at most 2 models" \
    --gpt_model "which GPT model to use for evaluation" \
    --save_path "path to save results" \
    --openai_key "your openai key" \
```

If you want GPT evaluation with reference, you can add an argument `--gpt_with_reference`, but make sure the reference file have target answers.

## FAQ

<details><summary><b>How can I add a new GPT evaluation metric?</b></summary>

For example, if you want to add a new metric `persuasiveness` into category `brainstorming`, you should add the metric definition and its corresponding CoT(Chain-of-thought) in the evaluation prompt file in `prompt/evaluation_promt`. The CoT can be generated using ChatGPT. You can prompt ChatGPT to generate evaluation steps for the new metric.

```json
{
  "brainstorming": {
    "id": 1,
    "category": "brainstorming",
    "metrics": {
      "persuasiveness": "persuasiveness(1-5)：a short description for persuasiveness"
    },
    "CoT": {
      "persuasiveness": "CoT for persuasiveness\n\npersuasiveness："
    },
    "prompt": "You are a good assistant. Please rate the given answer to the \"brainstorming\" question below.\n\nThe question is as follows:\n\n{question}\n\nThe answer is as follows:\n\n{answer}\n\nThe metric for evaluation is as follows:\n\n{metric}\n\nYou should follow the following evaluation steps:\n\n{steps}"
  }
}
```

</details>

## Citations

```bibtex
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://vicuna.lmsys.org},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}

@misc{liu2023geval,
      title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
      author={Yang Liu and Dan Iter and Yichong Xu and Shuohang Wang and Ruochen Xu and Chenguang Zhu},
      year={2023},
      eprint={2303.16634},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
