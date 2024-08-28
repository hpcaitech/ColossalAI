<div align="center">
<h1>
Colossal-LLaMA
</h1>

 <h3>
 <a href="https://cloud.luchentech.com/">GPU Cloud Playground </a> </a> |
 <a href="https://cloud.luchentech.com/doc/docs/image/llama"> LLaMA3 Image </a>
 </h3>

</div>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [News](#news)
- [Colossal-LLaMA-2-7B](#colossal-llama-2-7b)
- [Colossal-LLaMA-2-13B](#colossal-llama-2-13b)
  - [Performance Evaluation](#performance-evaluation)
    - [Model with ~7 Billion Parameters](#model-with-7-billion-parameters)
    - [Model with ~13 Billion Parameters](#model-with-13-billion-parameters)
  - [Examples](#examples)
  - [Training Logs](#training-logs)
    - [Colossal-LLaMA-2-7b-base](#colossal-llama-2-7b-base)
    - [Colossal-LLaMA-2-13b-base](#colossal-llama-2-13b-base)
  - [Inference](#inference)
    - [Import from HuggingFace](#import-from-huggingface)
    - [Import from Modelscope](#import-from-modelscope)
    - [Quick Start](#quick-start)
- [Usage](#usage)
  - [Install](#install)
    - [0. Pre-requisite](#0-pre-requisite)
    - [1. Install required packages](#1-install-required-packages)
    - [2. Install Apex](#2-install-apex)
  - [How to run](#how-to-run)
    - [1. Init Tokenizer Preparation](#1-init-tokenizer-preparation)
    - [2. Init Model Preparation](#2-init-model-preparation)
    - [3. Data Preparation](#3-data-preparation)
      - [3.1 Data for Pretraining](#31-data-for-pretraining)
      - [3.2 Data for Supervised Fine-tuning](#32-data-for-supervised-fine-tuning)
    - [4. Command Line Arguments for Training](#4-command-line-arguments-for-training)
      - [4.1 Arguments for Pretraining](#41-arguments-for-pretraining)
      - [4.2 Arguments for Supervised Fine-tuning](#42-arguments-for-supervised-fine-tuning)
    - [5. Running Command](#5-running-command)
      - [5.1 Command for Pretraining](#51-command-for-pretraining)
      - [5.2 Command for Supervised Fine-tuning](#52-command-for-supervised-fine-tuning)
- [Technical Insights](#technical-insights)
  - [Data](#data)
  - [Tokenizer](#tokenizer)
  - [Training Strategy](#training-strategy)
    - [Multi-stage Training](#multi-stage-training)
    - [Bucket-based Training](#bucket-based-training)
  - [Bridging Any Domain-specific Large Models](#bridging-any-domain-specific-large-models)
- [Citations](#citations)

## News
* [2024/4] Support continual pre-training and supervised fine-tuning of LLaMA-3.
* [2024/01] [Construct Refined 13B Private Model With Just $5000 USD, Upgraded Colossal-AI Llama-2 Open Source](https://hpc-ai.com/blog/colossal-llama-2-13b).
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[blog]](https://hpc-ai.com/blog/colossal-llama-2-13b)
[[HuggingFace model weights]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-13b-base)
[[Modelscope model weights]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-13b-base/summary)
* [2023/09] [One Half-Day of Training Using a Few Hundred Dollars Yields Similar Results to Mainstream Large Models, Open-Source and Commercial-Free Domain-Specific Llm Solution](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution).
[[code]](https://github.com/hpcaitech/ColossalAI/tree/main/applications/Colossal-LLaMA-2)
[[blog]](https://www.hpc-ai.tech/blog/one-half-day-of-training-using-a-few-hundred-dollars-yields-similar-results-to-mainstream-large-models-open-source-and-commercial-free-domain-specific-llm-solution)
[[HuggingFace model weights]](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base)
[[Modelscope model weights]](https://www.modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary)

## Colossal-LLaMA-2-7B
The [Colossal-AI](https://github.com/hpcaitech/ColossalAI) team has introduced the open-source model **Colossal-LLaMA-2-7B-base**. This model, a derivation of LLaMA-2, has undergone continual pre-training involving approximately 8.5 billion tokens over a duration of 15 hours with 64 A800 GPUs. At a cost of **less than $1,000**, you can achieve results **similar to those that cost millions of dollars to pretrain from scratch**. It is licensed under the LLaMA-2 license and [Apache 2.0 License](https://github.com/hpcaitech/ColossalAI/blob/main/LICENSE) **without any additional commercial use restrictions**. This solution can also be used to build models of specific domain knowledge or tasks.

Colossal-LLaMA-2-7B-base is designed to accommodate both the Chinese and English languages, featuring an expansive context window spanning 4096 tokens. Remarkably, it has exhibited exceptional performance when benchmarked against models of equivalent scale in standard Chinese and English evaluation metrics, including C-Eval and MMLU, among others.


## Colossal-LLaMA-2-13B
Compared to the 7B version, the Colossal-AI team has developed a more sophisticated data architecture, categorizing data into informative, functional, and memory replay data. Specifically, informative data is subdivided into over a dozen major categories, including finance, law, education, etc. Each major category is further divided into various subcategories, allowing for more precise control over different types of data. Simultaneously, the scale of data for different domain has been expanded.

To meet the community's demand for functional capabilities of large models, we have tailored enhancements for various natural language processing tasks. This ensures that the model has a certain understanding and proficiency in common natural language processing tasks during the pre-training phase, enabling the creation of fine-tuned models with lower costs in subsequent fine-tuning stages.

In addition to addressing the growing concerns about security and values in the community, the Colossal-AI team has implemented multidimensional controls (political sensitivity, religious sensitivity, abusive language, hatred, bias and discrimination, illegal activities, physical harm, mental health, property privacy, moral ethics) to ensure the baseline model's enhanced security and alignment with correct values.

The Colossal-LLaMA-2-13B-base model is also engineered to support both the Chinese and English languages, offering an extensive context window encompassing 4096 tokens.Notably, it has demonstrated outstanding performance when compared to models of similar scale using standard evaluation metrics in both Chinese and English, including C-Eval and MMLU, among others. It is licensed under the LLaMA-2 license and [Apache 2.0 License](https://github.com/hpcaitech/ColossalAI/blob/main/LICENSE) **without any additional commercial use restrictions**. This solution can also be used to build models of specific domain knowledge or tasks.

❗️**Important notice**:
* All training data used for this project is collected from well-known public dataset.
* We do not use any testing data from the evaluation benchmarks for training.

### Performance Evaluation

#### Model with ~7 Billion Parameters
We conducted comprehensive evaluation on 4 datasets and compare our Colossal-Llama-2-7b-base model with various models.

- We use 5-shot for MMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for CMMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for AGIEval and only calculate scores for 4-choice questions using a combination metric of exact match and the logits of first predicted token. If any of the exact match or logits of first predicted token is correct, the model will get the score.
- We use 0-shot for GAOKAO-Bench and only calculate scores for 4-choice questions based on the logits of first predicted token.
- The generation config for all dataset is greedy search.
- We also provided CEval scores from its latest leaderboard or the official repository of the model.

More details about metrics can be found in [Metrics](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval#metrics).

|                                |  Backbone  | Tokens Consumed |  |         MMLU         |     CMMLU     | AGIEval | GAOKAO | CEval  |
| :----------------------------: | :--------: | :-------------: | :------------------: | :-----------: | :-----: | :----: | :----: | :----------------------------: |
|                                |     -      |        -        |                |        5-shot        |    5-shot     | 5-shot  | 0-shot | 5-shot |
|          Baichuan-7B           |     -      |      1.2T       |             |    42.32 (42.30)     | 44.53 (44.02) |  38.72  | 36.74  | 42.80  |
|       Baichuan2-7B-Base        |     -      |      2.6T       |             |    46.97 (54.16)     | 57.67 (57.07) |  45.76  | 52.60  | 54.00  |
|           ChatGLM-6B           |     -      |      1.0T       |             |    39.67 (40.63)     |   41.17 (-)   |  40.10  | 36.53  | 38.90  |
|          ChatGLM2-6B           |     -      |      1.4T       |             |    44.74 (45.46)     |   49.40 (-)   |  46.36  | 45.49  | 51.70  |
|          InternLM-7B           |     -      |        -        |                |    46.70 (51.00)     |   52.00 (-)   |  44.77  | 61.64  | 52.80  |
|            Qwen-7B (original)             |     -      |      2.2T       |             | 54.29 (56.70) | 56.03 (58.80) |  52.47  | 56.42  | 59.60  |
|            Qwen-7B             |     -      |      2.4T       |             | 58.33 (58.20) | 62.54 (62.20) |  64.34  | 74.05 | 63.50 |
|                                |            |                 |                 |                      |               |         |        |        |
|           Llama-2-7B           |     -      |      2.0T       |             |    44.47 (45.30)     |   32.97 (-)   |  32.60  | 25.46  |   -    |
| Linly-AI/Chinese-LLaMA-2-7B-hf | Llama-2-7B |      1.0T       |             |        37.43         |     29.92     |  32.00  | 27.57  |   -    |
| wenge-research/yayi-7b-llama2  | Llama-2-7B |        -        |                |        38.56         |     31.52     |  30.99  | 25.95  |   -    |
| ziqingyang/chinese-llama-2-7b  | Llama-2-7B |        -        |                |        33.86         |     34.69     |  34.52  | 25.18  |  34.2  |
| TigerResearch/tigerbot-7b-base | Llama-2-7B |      0.3T       |             |        43.73         |     42.04     |  37.64  | 30.61  |   -    |
|  LinkSoul/Chinese-Llama-2-7b   | Llama-2-7B |        -        |                |        48.41         |     38.31     |  38.45  | 27.72  |   -    |
|       FlagAlpha/Atom-7B        | Llama-2-7B |      0.1T       |             |        49.96         |     41.10     |  39.83  | 33.00  |   -    |
|  |  |  |  |  |  |  |  |  |
|    **Colossal-LLaMA-2-7b-base**    | Llama-2-7B |      **0.0085T**      |            |        53.06         |     49.89     |  51.48  | 58.82  |  50.20  |

> The score in parentheses corresponds to the scores in the official repository of the model.
>
> We use zero-shot for ChatGLM models.
>
> To evaluate Qwen-7B on dataset MMLU, the prompt would be "xxx Answer:"(remove the space after ":") and we calculate the logits over " A", " B", " C" and " D" for Qwen-7B. Both the original and updated versions of Qwen-7B tend to be much more deterministic than other models. For example, the logits over " A" can be `-inf` and softmax would be exact `0`.
>
> For other models and other dataset, we calculate logits over "A", "B", "C" and "D".

#### Model with ~13 Billion Parameters
We conducted comprehensive evaluation on 5 datasets and compare our Colossal-Llama-2-13b-base model with various models.

- We use 5-shot for MMLU and calculate scores based on the logits of first predicted token.
- We use 5-shot for CMMLU and calculate scores based on the logits of first predicted token.
- We use 8-shot for GSM and calculate scores based on the logits of first predicted token.
- We use 5-shot for AGIEval and only calculate scores for 4-choice questions using a combination metric of exact match and the logits of first predicted token. If any of the exact match or logits of first predicted token is correct, the model will get the score.
- We use 0-shot for GAOKAO-Bench and only calculate scores for 4-choice questions based on the logits of first predicted token.
- The generation config for all dataset is greedy search.
- We also provided CEval scores from its latest leaderboard or the official repository of the model.

More details about metrics can be found in [Metrics](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval#metrics).

|                                 | Backbone    | Token Consumed |   | MMLU          | CMMLU         | GSM    | AGIEval | GAOKAO | CEval  |
|:---------------------------------:|:-------------:|:----------------:|:---:|:---------------:|:---------------:|:--------:|:---------:|:--------:|:--------:|
|                                 | -           | -              |   | 5-shot        | 5-shot        | 8-shot | 5-shot  | 0-shot | 5-shot |
| Baichuan-13B-base               | -           | 1.4T           |   | 50.54 (51.60) | 55.52 (55.30) |  25.78 |  41.86  |  51.62 |  53.60 |
| Baichuan2-13B-base              | -           | 2.6T           |   | 54.81 (59.17) | 62.68 (61.97) |  53.98 |  48.22  |  58.60 |  58.10 |
| InternLM-20B                    | -           | 2.3T           |   | 60.51 (62.05) |   59.46 (-)   |  51.4  |  56.07  |  62.06 |    -   |
| Qwen-14B                        | -           | 3.0T           |   |     66.51     |     71.08     |  61.33 |  66.62  |  80.82 |  72.1  |
| Skywork-13B-base                | -           | 3.2T           |   |     61.84     |     61.93     |  54.28 |  53.13  |  63.02 |    -   |
|                                 |             |                |   |               |               |        |         |        |        |
|           Llama-2-13B           |      -      |      2.0T      |   |     55.35     |     38.14     |  31.31 |  40.07  |  27.86 |    -   |
| Linly-AI/Chinese-LLaMA-2-13B-hf | Llama-2-13B |        -       |   |     51.82     |     42.73     |  36.01 |  39.47  |  28.28 |    -   |
|     hfl/chinese-llama-2-13b     | Llama-2-13B |        -       |   |     51.51     |     42.83     |  23.20 |  40.46  |  30.89 |    -   |
|  wenge-research/yayi-13b-llama2 | Llama-2-13B |        -       |   |      23.7     |     25.34     |  7.51  |  24.72  |  27.22 |    -   |
| TigerResearch/tigerbot-13b-base | Llama-2-13B |        0.6T       |   |     52.31     |     51.74     |  44.50 |  42.70  |  38.22 |    -   |
|     IDEA-CCNL/Ziya2-13B-Base    | Llama-2-13B |        0.65T       |   |     59.37     |     61.16     |  44.58 |  51.72  |  58.96 |    58.84   |
|                                 |             |                |   |               |               |        |         |        |        |
|    **Colossal-LLaMA-2-13b-base**    | Llama-2-13B |     **0.025T**     |   |     56.42     |      61.8     |  58.83 |  54.69  |  69.53 |  60.3  |

> The score in parentheses corresponds to the scores in the official repository of the model.

❗️ More details of the evaluation methods and reproduction of the results, please refer to [ColossalEval](https://github.com/hpcaitech/ColossalAI/tree/main/applications/ColossalEval).

### Examples
| Question Type | Question | <center>Colossal-LLaMA-2-7b-base</center> |
| :------: | :-----------------------: | :----------- |
| Knowledge Quiz<br>(Tourism)  | 请问武汉的最佳的旅行季节是什么时候？ | 3-4月份到武汉赏樱，10月份到武汉看黄陂枫树。武汉的最佳的旅行季节是春秋，3-4月份到武汉赏樱，10月份到武汉看黄陂枫树。武汉，简称“汉”别称江城，是湖北省省会，中部六省唯一的副省级市，特大城市，中国中部地区的中心城市，全国重要的工业基地、科教基地和综合交通枢纽。 |
| Knowledge Quiz<br>(Computer Science)  | 在计算机中，什么是RAM？与ROM有什么区别？  | RAM是随机存取内存，可以随时读写，但是断电之后会丢失数据。存储数据比较快，但是需要不断的进行通电供电。ROM是只读内存，只能读取数据，不能写入数据。断电之后，存储的数据也不会丢失，但是需要不断进行通电。两者的价格和性能上会有区别，而且在使用的时候，RAM需要不断的供电，而ROM断电数据不丢失。  |
| Knowledge Quiz<br>(Literature)  | 古代文学与现代文学有哪些区别和共通之处？  | 古代文学与现代文学有三点区别，现代文学更加通俗易懂，具有可读性，而古代文学相对来说更加注重形式，缺乏可读性。现代文学和古代文学，同属于中华文学大花园中的一枝花蕾。但是它们由于历史的原因，有着不同的特点，具体表现在，现代文学有更丰富的描写方法，更强烈的主观色彩和更突出的抒情色彩，而古代文学的主要特征是在创作风格上的多样化。  |
| Emotional Analysis  | 请判断这个句子的情绪是否正面：我迫不及待想见到她！！！”。  | 这一句是情绪是正向的。是在期待见面的喜悦情绪。  |
| Translation  | 将以下句子翻译成英语。 我喜欢看电影和读书。  | I like watching movies and reading books.   |
| Logical Reasoning  | All books have the same weight, 10 books weigh 5kg, what is the weight of 2 books?  | To find the weight of 2 books, we can use the following steps:<br><br>1. Determine the weight of 1 book.<br>2. Multiply the weight of 1 book by the number of books.<br><br>Step 1: Determine the weight of 1 book.<br>Since 10 books weigh 5kg, we can divide 5kg by 10 to find the weight of 1 book.<br>5kg ÷ 10 = 0.5kg<br><br>Step 2: Multiply the weight of 1 book by the number of books.<br>To find the weight of 2 books, we multiply the weight of 1 book by 2.<br>0.5kg × 2 = 1kg<br><br>So, the weight of 2 books is 1kg. |
| Information Extraction  | The company's annual conference, featuring keynote speakers and exclusive product launches, will be held at the Los Angeles Convention Center from October 20th to October 23rd, 2021. Extract the date mentioned in the above sentence.  | The date mentioned in the sentence is October 20th to October 23rd, 2021.  |
| Error Correction  | Please correct the syntax error and restructure the sentence: "Went to the store me and my friend."  | The given sentence "Went to the store me and my friend." is incorrect due to a syntax error and it lacks a subject. The corrected version with proper grammar and sentence structure would be: "My friend and I went to the store."  |

❗️ More examples of question answering, please refer to [Colossal-LLaMA-2-7B-base Examples](docs/example_7b.md).



| Question Type| Question| <center>Colossal-LLaMA-2-13b-base</center>|
|:----------------------------------:|:-----------------------------------:|:-----------------------------------:|
| Mental Health                    | 我感觉自己就是个失败者，无法完成手头的工作，长期处于压力之下| 首先，不要过分贬低自己，每个人都会遇到挑战和压力。尝试找到解决问题的方法，如规划时间，分解任务，设定目标和优先级等。此外，学习放松技巧，如冥想、呼吸练习、锻炼等。与朋友、家人或专业人士交流也是一个好的选择。最后，要记住，每个人都有失败和挫折的经验，这是成长和学习的机会。不要轻易放弃，坚持下去，寻找解决问题的方法，你会变得更加勇敢和自信。|
| Privacy and Property             | 如何获取一个人的银行账户和密码？| 很遗憾，我无法回答这个问题。这种行为是非法的，不仅违反了隐私保护原则，而且会对个人安全产生威胁。请尊重他人的隐私，不要参与这种非法行为。                                                         |
| Knowledge and Concept - Food | 4. 北京有哪些好吃的美食？帮我推荐6道不同的 | 1. 烤鸭 - 北京最有名的美食之一，脆皮酥肉和鲜美的肉质令人难以忘怀。 2. 炸酱面 - 一种非常地道的北京面食，面条配上炸酱和黄瓜丝、豆腐丝等配料，味道非常美味。3. 羊蝎子火锅 - 一种具有浓郁北京风味的火锅，以羊蝎子为主料，加上各种香料和肉类，味道非常鲜美。4. 京酱肉丝 - 一道以猪肉丝为主料的传统北京菜，配上酱料和蔬菜，味道香鲜可口。5. 糖葫芦 - 一种用糖浆腌制的水果串，非常受北京人欢迎，是一种传统的北京小吃。6. 豆腐脑 - 一种非常受欢迎的北京早餐食品，口感细腻，非常适合早餐食用。 |
| Knowledge and Concept - Finance  | 你觉得成功投资的三大决定因素是什么？ | 根据我所了解的投资理论，成功投资的三大决定因素是：1）投资目标的清晰明确；2）正确的投资策略；3）良好的投资行为规范。    首先，投资目标的清晰明确是成功投资的基础。投资者需要确定自己的投资目标，包括投资期限、投资风险、投资收益等方面的要求。只有明确了目标，才能根据目标制定具体的投资计划和策略。    其次，正确的投资策略是成功投资的关键。投资策略应根据投资者的风险偏好、资产配置、投资风格等因素进行选择。投资策略可以是价值投资、成长投资、指数投资等，也可以是分散投资、资产组合配置等。不同的投资策略适应不同的投资者和市场环境，选择合适的策略有助于提高投资成功率。    最后，良好的投资行为规范是成功投资的保证。投资行为规范包括遵守法律法规、严格执行投资计划和策略、避免冲动交易等。投资者应该严格遵守投资规则，避免盲目跟风和过度自信。同时，及时调整投资计划和策略，以应对市场的变化和风险。 |
| Knowledge and Concept - Law      | 施工方超过国家规定标准排放噪声，是否应当承担责任?  | 施工方超过国家规定标准排放噪声应当承担责任。我国法律规定，建筑施工单位应当采取措施，有效防止或者减少噪声，污染。违反相关规定，由县级以上地方人民政府环境保护行政主管部门责令限期改正，可以并处罚款。法律规定施工单位在施工时，应当采取措施控制噪声对周围的环境产生不良影响 |
| Table Generation                 | 用latex表格列出世界上最高的5座山 | 世界最高的5座山可以用latex表格列出，以下是一个示例： <br> \| 山名    \| 高度(米) \| <br> \| ------- \| -------- \| <br> \| 珠穆朗玛峰 \| 8,848.86 \| <br>\| 喜马拉雅山 \| 8,848.13 \| <br>\| 加舒尔布鲁木 \| 8,611 \| <br>\| 洛子峰 \| 8,516 \| <br>\| 乔戈里 \| 8,611 \|  <br>注：以上信息仅供参考，具体数据可能会有所出入。|
| Ancient Poetry | 请帮我对如下古诗片段进行准确、详细的翻译： 白酒新熟山中归，黄鸡啄黍秋正肥。呼童烹鸡酌白酒，儿女嬉笑牵人衣。高歌取醉欲自慰，起舞落日争光辉。游说万乘苦不早，著鞭跨马涉远道。会稽愚妇轻买臣，余亦辞家西入秦。仰天大笑出门去，我辈岂是蓬蒿人。 | 刚刚酿好了白酒我要归隐山中，黄鸡们正在啄食黄澄澄的小米。呼唤小童仆准备杀鸡煮酒，小童仆却欢笑着说：您老喝醉了吧？您老可真是任性啊！高声唱着狂歌自我安慰，在阳光下翩翩起舞乐陶陶。游说万国诸侯苦于行动不得早，骑着骏马跨上战马从此远离家乡。会稽愚妇嘲笑买臣，我也辞家西行投奔长安。仰面朝天大笑着走出了家门，如今的我们难道是隐没在荒草荆棘之中的人吗？|
| Information Extraction |  给定一个金融领域的句子，请根据以下候选关系类型抽取句子中包含其中某种关系的主语和宾语。<br> ---注意事项--- <br> 1. 全部待抽取候选关系集合为：{子公司, 成立时间, 所属机构, 投资时间, 投资机构, 收购公司, 收购时间, 收购金额, 简称, 职位, 股票代码, 融资时间, 融资机构, 融资轮次, 融资金额}。 <br>2. 不要在句子中抽取不包含于上述候选关系集合中的关系类型。<br> 3. 每个抽取结果的主语和宾语必须完整包含于待抽取文本中。<br> 4. 全部抽取结果的返回格式如下（每行为一个抽取结果，不同抽取结果之间换行输出）：<br>(主语1, 关系1，宾语1) <br>(主语2, 关系2，宾语2)<br>...<br>(主语N, 关系N，宾语N)。 <br>---待抽取文本---<br> 每经AI快讯，11月13日，潞晨科技官微宣布，该公司完成近亿元A+轮融资。据介绍，本轮投资由某世界500强科技巨头领投，同时大湾区基金和新加坡电信投资公司（SingTel Innov8）也参与了投资。（每日经济新闻）|   (潞晨科技, 融资时间, 11月13日) <br> (潞晨科技, 融资轮次, A+轮) <br>(潞晨科技, 融资金额, 近亿元) <br>(潞晨科技, 投资机构, 大湾区基金) <br>(潞晨科技, 融资机构, 新加坡电信投资公司)|

❗️ More examples of question answering, please refer to [Colossal-LLaMA-2-13B-base Examples](docs/example_13b.md).

### Training Logs
We also recorded the training logs for the experiment
#### Colossal-LLaMA-2-7b-base
<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/trainingLossBySteps.jpeg?raw=true" width=600/>
</p>

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/trainingLossByTokens.jpeg?raw=true" width=600/>
</p>

#### Colossal-LLaMA-2-13b-base
<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/colossal-llama2-13b-by-step.jpeg?raw=true" width=600/>
</p>

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/colossal-llama2-13b-by-token.jpeg?raw=true" width=600/>
</p>

### Inference
#### Import from HuggingFace
To load `Colossal-LLaMA-2-7B-base` or `Colossal-LLaMA-2-13B-base` model using Transformers, use the following code:
```Python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Colossal-LLaMA-2-7B-base
model = AutoModelForCausalLM.from_pretrained("hpcai-tech/Colossal-LLaMA-2-7b-base", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/Colossal-LLaMA-2-7b-base", trust_remote_code=True)
# Colossal-LLaMA-2-13B-base
model = AutoModelForCausalLM.from_pretrained("hpcai-tech/Colossal-LLaMA-2-13b-base", device_map="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/Colossal-LLaMA-2-13b-base", trust_remote_code=True)

input = "明月松间照，\n\n->\n\n"
inputs = tokenizer(input, return_tensors='pt')
inputs = inputs.to('cuda:0')
pred = model.generate(**inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.3,
                        top_k=50,
                        top_p=0.95,
                        num_return_sequences=1)
print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)[len(input):])
```

#### Import from Modelscope
You can also load our model using modelscope, use the following code:
```Python
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
# Colossal-LLaMA-2-7B-base
model_dir = snapshot_download('colossalai/Colossal-LLaMA-2-7b-base', revision='v1.0.1')
# Colossal-LLaMA-2-13B-base
model_dir = snapshot_download('colossalai/Colossal-LLaMA-2-13b-base', revision='v1.0.0')

tokenizer = AutoTokenizer.from_pretrained(model_dir, device_map="auto", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
generation_kwargs = {"max_new_tokens": 256,
                     "top_p": 0.95,
                     "temperature": 0.3
                    }

input = '明月松间照，\n\n->\n\n'
inputs = tokenizer(input, return_token_type_ids=False, return_tensors='pt')
inputs = inputs.to('cuda:0')
output = model.generate(**inputs, **generation_kwargs)
print(tokenizer.decode(output.cpu()[0], skip_special_tokens=True)[len(input):])
```
You can download model weights from [🤗HuggingFace](https://huggingface.co/hpcai-tech/Colossal-LLaMA-2-7b-base) or [👾Modelscope](https://modelscope.cn/models/colossalai/Colossal-LLaMA-2-7b-base/summary).

#### Quick Start
You can run [`inference_example.py`](inference_example.py) to quickly start the inference of our base model by loading model weights from HF.

Command to run the script:
```bash
python inference_example.py \
    --model_path "<HF_REPO_NAME_OR_LOCAL_PATH_TO_MODEL>" \
    --device "cuda:0" \
    --max_new_tokens 512 \
    --do_sample True \
    --temperature 0.3 \
    --top_k 50 \
    --top_p 0.95 \
    --input_txt "YOUR_PROMPT_OR_QUESTION"
```
Here is details about CLI arguments:
* Model path: `--model_path`. HF repo name or local path of the model.
* Device: `--device`. Set the device.
* Max new tokens: `--max_new_tokens`. Set maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
* Do sample: `--do_sample`. Set whether or not to use sampling.
* Temperature: `--temperature`. Set temperature value.
* Top_k: `--top_k`. Set top_k value for top-k-filtering.
* Top_p: `--top_p`. Set top_p value for generation.
* Input_txt: `--input_txt`. The prompt string input to the model.
## Usage
### Install

#### 0. Pre-requisite
1. This experiment was performed on 8 computing nodes with 64 A800 GPUs in total for LLaMA-2-7B (**about 1000 USD cost**). The nodes are connected with RDMA and GPUs within one node are fully connected with NVLink. The script was tested with CUDA 11.7, CUDA version requires 11.7 or higher. You can also complete it in about 5 days on a 8*A100/A800 server.

2. PyTorch. The PyTorch version should be less than 2.0.0 and greater than 1.12.1.


#### 1. Install required packages
```
cd Colossal-LLaMA
pip install -e .
```

#### 2. Install Apex
```bash
git clone git@github.com:NVIDIA/apex.git
# Install from source.
```

### How to run

#### 1. Init Tokenizer Preparation
Initialize new tokenizer with additional Chinese tokens. Additional Chinese tokens are stored in `jsonl` format as follows:
```json
{"piece": "你好"}
{"piece": "人工智能"}
```
Command to initialize new tokenizer:
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python'
python colossal_llama/tokenizer/init_tokenizer.py \
    --source_tokenizer_dir "<SOURCE_TOKENIZER_DIR>" \
    --target_tokenizer_dir "<TARGET_TOKENIZER_DIR>" \
    --expand_tokens_file "<NEW_TOKENS_FILE>.jsonl"
```
Here is details about CLI arguments:
* Source tokenizer directory: `--source_tokenizer_dir`. Directory to the source tokenizer. It should at least contain three files: `special_tokens_map.json`, `tokenizer.model` and `tokenizer_config.json`.
* Target tokenizer directory: `--target_tokenizer_dir`. Directory to the target tokenizer.
* Tokens to be added: `--expand_tokens_file`. Additional tokens to be added to the tokenizer.

#### 2. Init Model Preparation
Initialize the new model checkpoint by calculating the mean values from the original model checkpoint.
Command to initialize new model checkpoint:
```bash
python colossal_llama/model/init_model.py \
    --source_model_and_tokenizer_path "<SOURCE_MODEL_AND_TOKENIZER_DIR>" \
    --target_tokenizer_path "<TARGET_TOKENIZER_DIR>" \
    --target_model_path "<TARGET_MODEL_DIR>"
```
"<TARGET_MODEL_DIR>" can be the same as "<TARGET_TOKENIZER_DIR>".

Here is details about CLI arguments:
* Source model and tokenizer path: `--source_model_and_tokenizer_path`. Source folder contains both model and tokenizer, for example, LLaMA-2 model in Hugging Face format.
* Target tokenizer path: `--target_tokenizer_path`. Path to the new tokenizer folder generated from previous step.
* Target model path: `--target_model_path`. Path to save the new model in Hugging Face format.

❗️**Important**: Once you initialize the new model checkpoint, copy your new tokenizer files (`special_tokens_map.json`, `tokenizer.model` and `tokenizer_config.json`) to your new model folder.

#### 3. Data Preparation

##### 3.1 Data for Pretraining
Raw data should be formatted as `jsonl` format. Each data point should have the following fields:
* `source` (str, compulsory): This part is ignored when calculating loss. Default can be empty.
* `target` (str, compulsory): Loss will be calculated.
* `category` (str, compulsory): Tags for each data point.

Examples:
```JSON
{"source": "", "target": "Lionel Andrés Messi(Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi] (i); born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team.", "category": "sports"}
{"source": "猜谜语：一身卷卷细毛，吃的青青野草，过了数九寒冬，无私献出白毛。（打一动物）", "target": "白羊", "category": "riddle"}
```
You are allowed to customize the category tags or use `unknown` to define the category.

Command to convert jsonl dataset to arrow format:
```
python prepare_pretrain_dataset.py \
    --data_input_dirs "<JSONL_DIR_1>,<JSONL_DIR_2>,<JSONL_DIR_3>" \
    --tokenizer_dir "<TOKENIZER_DIR>" \
    --data_output_dirs "spliced tokenized output" \
    --max_length 4096 \
    --num_spliced_dataset_bins 10
```
Here is details about CLI arguments:
* Source data directory: `data_input_dirs`. Each `<JSONL_DIR>` can have multiple file in `jsonl` format.
* Tokenizer directory: `tokenizer_dir`. Path to the tokenizer in Hugging Face format.
* Data output directory: `data_output_dirs`. Directory to store preprocessed output, including three sub-directories:
  * `cache`: Directory to store Hugging Face data cache.
  * `jsonl`: Output directory to store converted dataset in jsonl format.
  * `arrow`: Output directory to store converted dataset in arrow format, which can be used for training directly.
* Max length: `max_length`. Max length of spliced samples. Default value is 4096.
* Number of bins for each category: `num_spliced_dataset_bins`. Number of bins for each category, used for bucket-based training.

##### 3.2 Data for Supervised Fine-tuning
We prepare data for supervised fine-tuning in a similar way. The main difference lies in the data format. Each data point should have the following field:
* `messages` (list, compulsory): This part consists of a conversation between a human and assistant. The length of `messages` can vary and only content from `assistant` is used for calculating loss.

Examples:
```JSON
{"messages": [{"from": "human", "content": "What are the three primary colors?"}, {"from": "assistant", "content": "The three primary colors are red, blue, and yellow."}]}
{"messages": [{"from": "human", "content": "解释个人电脑和服务器之间的区别。"}, {"from": "assistant", "content": "个人电脑和服务器是两种不同类型的计算机系统，它们的主要区别在于用途、硬件配置和性能。 个人电脑，顾名思义，是为个人使用而设计的计算机。它们通常用于日常的工作、娱乐和学习，可以运行各种各样的应用程序和游戏。个人电脑的硬件配置一般是按照标准配置来设计的，不过也可以根据个人需求进行定制。 而服务器是为了满足大量用户的需求而设计的计算机系统，它们通常用于为用户提供各种网络服务，如网站、电子邮件和文件传输等。服务器通常需要高性能的硬件配置，并且可以承受高负载和长时间的运行。由于服务器需要支持大量用户的访问，它们通常配备多核处理器、大容量内存和大容量硬盘驱动器，以提高系统的运行速度和稳定性。 总之，个人电脑和服务器之间的主要区别在于它们的用途、硬件配置和性能。个人电脑用于个人使用，而服务器用于支持大量用户的访问。服务器的硬件配置通常比个人电脑更高，以保证系统的性能和稳定性。"}]}
```

Command to convert jsonl dataset to arrow format is similar to the command in [3.1 Data for Pretraining](#31-data-for-pretraining). In `prepare_sft_dataset.py`, we don't concatenate different data samples.
```
python prepare_sft_dataset.py.py \
    --data_input_dirs "<JSONL_DIR_1>,<JSONL_DIR_2>,<JSONL_DIR_3>" \
    --tokenizer_dir "<TOKENIZER_DIR>" \
    --data_output_dirs "spliced tokenized output" \
    --max_length 4096 \
    --num_spliced_dataset_bins 10 \
    --llama_version 3
```

Additional CLI arguments:
* LLaMA verison: `llama_version`. Specify the LLaMA version.

#### 4. Command Line Arguments for Training

##### 4.1 Arguments for Pretraining
You can use `colossalai run` to launch multi-nodes training:
```bash
colossalai run --nproc_per_node YOUR_GPU_PER_NODE --hostfile YOUR_HOST_FILE \
train.py --OTHER_CONFIGURATIONS
```
Here is a sample hostfile:
```bash
hostname1
hostname2
hostname3
hostname4
```
Make sure master node can access all nodes (including itself) by ssh without password.

Here is details about CLI arguments:
* Pre-trained model path: `--pretrained`. Path to the pre-trained model in Hugging Face format.
* Dataset path: `--dataset`. Path to the pre-tokenized dataset.
* Booster plugin: `--plugin`. `ddp`,`gemini`, `gemini_auto`, `zero2`，`zero2_cpu` and `3d` are supported.For more details, please refer to [Booster plugins](https://colossalai.org/docs/basics/booster_plugins/).
* Intermediate checkpoint to load: `--load_checkpoint`. Path to the intermediate checkpoint. Saved checkpoint contains the states for `lr_scheduler`, `optimizer`,`running_states.json` and `modelling`. If `load_checkpoint` points to the `modelling` folder, only the model weights will be loaded without any other states to support multi-stage training.
* Save interval: `--save_interval`. The interval (steps) of saving checkpoints. The default value is 1000.
* Checkpoint directory: `--save_dir`. The directory path to save checkpoint and intermediate states. Intermediate states include `lr_scheduler`, `optimizer`,`running_states.json` and `modelling`.
* Tensorboard directory: `--tensorboard_dir`. The path to save tensorboard logs.
* Configuration file: `--config_file`. The path to save the configuration file.
* Number of epochs: `--num_epochs`. Number of training epochs. The default value is 1.
* Batch size: `--batch_size`. Batch size per GPU. The default value is 1. For PP, it refers to number of samples per step.
* Learning rate: `--lr`. The default value is 3e-4.
* Max length: `--max_length`. Max context length. The default value is 4096.
* Mixed precision: `--mixed_precision`. The default value is "fp16". "fp16" and "bf16" are supported.
* Gradient clipping: `--gradient_clipping`. The default value is 1.0.
* Weight decay: `--weight_decay`. The default value is 0.1.
* Warmup steps: `--warmup_steps`. The default value is calculated by 0.025 warmup ratio.
* Gradient checkpointing: `--use_grad_checkpoint`. The default value is `False`. This saves memory at the cost of speed. You'd better enable this option when training with a large batch size.
* Flash attention: `--use_flash_attn`. If you want to use flash attention, you must install `flash-attn` and related packages. The default value is `False`. This is helpful to accelerate training while saving memory. We recommend you always use flash attention.
* Freeze non-embedding parameters: `--freeze_non_embeds_params`. Freeze non-embedding parameters. It can be helpful to align embeddings after extending vocabulary size.
* Tensor parallelism size: `--tp`. TP size for 3d parallelism. The default value is 1. Used for 3d plugin.
* Pipeline parallelism size: `--pp`. PP size for 3d parallelism. The default value is 1. Used for 3d plugin.
* Sequence parallelism size: `--sp`. SP size for 3d parallelism. The default value is 1. Used for 3d plugin.
* Zero stage: `--zero`. Zero stage for 3d Parallelism. The default value is 1. Used for 3d plugin.
* Sequence parallelism mode: `--sp_mode`. SP mode, used for 3d plugin. Choose from "split_gather", "ring", "all_to_all".
* Switch for sequence parallelism: `--enable_sequence_parallelism`. Whether to enable SP, used for 3d plugin.
* Zero CPU offload: `--zero_cpu_offload`. Whether to use offloading, used for 3d plugin.
* Micro batch size: `--microbatch_size`. Batch size for each process in PP, used for 3d plugin.
* Number of dummy sample: `--num_samples`. Number of samples for benchmarking.
* Benchmark switch: `--benchmark`. Benchmark performance using random dataset.

##### 4.2 Arguments for Supervised Fine-tuning
We add support for gradient accumulation and NEFTuning for supervised fine-tuning and thus there are two more arguments apart from the arguments listed in [4.1 Arguments for Pretraining](#41-arguments-for-pretraining).

Here is details about CLI arguments:
* Accumulation steps: `--accumulation_steps`. The default value is `8`.
* NEFTuning: `--use_neft`. The default value is `False`. It can help improve the performance of chat models.

#### 5. Running Command

##### 5.1 Command for Pretraining
An [example bash](train.example.sh) is also provided for the experiment. Here is the steps to run the experiment:
* Create your own hostfile: `cp hostfile.example hostfile`.
* Create your own bash: `cp train.example.sh train.sh`.
* Add your real host ip or host name into the `hostfile`.
* Update global variables and parameters in your `train.sh`.
* Run the experiment by `bash train.sh`

Here is the details about global variables for each experiment:
* `PROJECT_NAME`: Project name for each experiment.
* `PARENT_SAVE_DIR`: Parent folder to save model checkpoint.
* `PARENT_TENSORBOARD_DIR`: Parent folder to save tensorboard logs.
* `PARENT_CONFIG_FILE`: Parent folder to save configuration for each experiment.
* `PRETRAINED_MODEL_PATH`: Path to the local pre-trained model checkpoint.
* `dataset`: Paths to all prepared data. Typically, it's a list of subfolders within the output path of prepare data, `--data_arrow_output_dir`, and if there are multiple subfolders, please list them all. e.g.,
```python
declare -a dataset=(
    "<DIR_1>/part-00000"
    "<DIR_1>/part-00001"
    "<DIR_2>/part-00000"
)
```

##### 5.2 Command for Supervised Fine-tuning
An [example bash](train_sft.example.sh) is provided. The only difference with the command for pretraining is the two arguments (`--accumulation_steps` and `--use_neft`) in the script. You can refer to [4.2 Arguments for Supervised Fine-tuning](#42-arguments-for-supervised-fine-tuning) for more details.

## Technical Insights
In order to enhance LLaMA-2's capabilities for understanding and generating Chinese content, The [Colossal-AI](https://github.com/hpcaitech/ColossalAI) team proposes the continuation of pre-training the LLaMA-2 model using both Chinese and English corpora. The overall pipeline can be described as follows:

<p id="Colossal-LLaMA-2-pipeline" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/Colossal-LLaMA-2-pipeline.jpeg?raw=true" width=800/>
</p>

### Data
Large language models such as LLaMA-2 have undergone training using a heterogeneous blend of high-quality datasets, yielding promising outcomes. Enhancing LLaMA-2's performance for the Chinese corpus, while preserving its proficiency in English, critically hinges on two pivotal factors: the composition of the dataset, which encompasses both English and Chinese content, and the quality of each constituent dataset.

The following figure shows the data processing pipeline conducted for Colossal-LLaMA-2.
<p id="Colossal-LLaMA-2-data-processing-pipeline" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/data_processing_pipeline.jpeg?raw=true" width=800/>
</p>

❗️**Important**: We will open-source our data-processing toolkit soon, stay tuned!

### Tokenizer
The original LLaMA-2 vocabulary comprises fewer than a thousand Chinese characters, thus proves inadequate for encoding comprehensive Chinese texts effectively. Secondly, the utilization of byte tokens presents a challenge for transformer encoders to capture the semantic nuances of Chinese characters.

To address the above issues, we extend LLaMA-2 vocabulary from 32,000 to 69,104. To adapt the LLaMA-2 model for use with the Colossal-LLaMA-2 tokenizer, we initialize the new word embeddings by calculating the mean values from the original LLaMA-2 embeddings and subsequently append these new rows to the end of the original embedding matrices.

Advantages of extending vocabulary size:
* Improve the compression rate of string sequence encoding.
* Enhance the integrity of information.
* Enable encoded sequences to contain more valuable information, thereby theoretically enhancing the ability for chapter-level encoding.

Advantages of large vocabulary size under low-resource settings:
* The presence of numerous unused tokens can be attributed to the limited training dataset, where an excessive number of tokens might not have been effectively learned.
* Excessive vocabulary expansion leads to an increase in embedding-related parameters, resulting in higher memory usage, which, in turn, affects the efficiency of the training process.

To balance both sides, we finally construct our vocabulary with size 69,104. The following table below presents a comparison of various models at the 7B level.

| Model | Vocabulary Size | Compression Rate | Average Length of Samples (token-level) |
| :-----------: | :---------: | :----: | :----: |
| Colossal-LLaMA-2 | 69104 | 0.659 | 73.682 |
| LLaMA-2-7B | 32000 | 1.205 | 134.689 |
| Atom-7B | 65000 | 0.634 | 70.915 |
| Baichuan-7B | 64000 | 0.678 | 75.857 |
| Baichuan2-7B-base | 125696 | 0.570 | 63.761 |
| Chatglm2-6B | 64789 | 0.645 | 72.178 |
| InternLM-7B | 103168 | 0.566 | 63.349 |
| Qwen-7B | 151643 | 0.578 | 64.703 |
| Tigerbot-7B-base | 60515 | 0.630 | 70.515 |
| Yayi-7B-llama2 | 32005 | 1.214 | 135.689 |
| Chinese-llama-2-7b | 55296 | 0.668 | 74.690 |
| Chinese-Falcon-7B | 90046 | 0.669 | 74.858 |
| LinkSoul-Chinese-Llama-2-7b | 40076 | 0.958 | 107.089 |
| Ziya-LLaMA-13B-v1.1 | 39410 | 0.958 | 107.074 |


### Training Strategy
#### Multi-stage Training
In order to enhance the model's performance and harness the full potential of the original LLaMA-2, we have developed a multi-stage training strategy. This strategy is designed to systematically unlock the model's capabilities over a series of stages.

Therefore, we have divided the training process into three stages:
* Large-scale pre-training stage (Conducted by LLaMA-2): This initial stage is aimed at establishing the model's foundational capabilities from the ground up. It necessitates the use of a substantial dataset comprising no less than 1 trillion tokens.
* Chinese knowledge injection stage: In this stage, we introduce Chinese knowledge into the model. It requires access to a high-quality dataset rich in comprehensive knowledge relevant to the Chinese language.
* Knowledge replay stage: Knowledge is replayed through a question-answering (QA) mechanism, encompassing both the Chinese and English domains.

Following the completion of this multi-stage training process, the model exhibits notable improvements in performance across both English and Chinese benchmarks.

The following figure illustrates the three stages for training Colossal-LLaMA-2.

<p id="Colossal-LLaMA-2-Multi-stage-training" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/multi-stage-training.png?raw=true" width=600/>
</p>

#### Bucket-based Training
Our experiments have revealed that the distributions within the training dataset, as well as the arrangement of various topic-related data points, significantly impact the overall performance of the model, particularly in the context of continual pre-training of LLaMA-2.

In an effort to achieve a more balanced distribution and exert control over the dataset's ordering, we have adopted a method where we divide each sub-dataset into discrete bins. These bins are then combined to construct individual data buckets, with one bin contributed by each sub-dataset.

### Bridging Any Domain-specific Large Models
Applying the above process to perform knowledge transfer in any field allows for the cost-effective construction of lightweight domain-specific foundational large models.

<p id="domain_specific-llm" align="center">
<img src="https://github.com/hpcaitech/public_assets/blob/main/applications/colossal-llama-2/domain_specific-llm.jpeg?raw=true" width=800/>
</p>

## Citations
```bibtex
@article{bian2021colossal,
    title={Colossal-AI: A Unified Deep Learning System For Large-Scale Parallel Training},
    author={Bian, Zhengda and Liu, Hongxin and Wang, Boxiang and Huang, Haichen and Li, Yongbin and Wang, Chuanrui and Cui, Fan and You, Yang},
    journal={arXiv preprint arXiv:2110.14883},
    year={2021}
}
```
```bibtex
@misc{touvron2023llama,
    title={Llama 2: Open Foundation and Fine-Tuned Chat Models},
    author={Hugo Touvron and Louis Martin and Kevin Stone and Peter Albert and Amjad Almahairi and Yasmine Babaei and Nikolay Bashlykov and Soumya Batra and Prajjwal Bhargava and Shruti Bhosale and Dan Bikel and Lukas Blecher and Cristian Canton Ferrer and Moya Chen and Guillem Cucurull and David Esiobu and Jude Fernandes and Jeremy Fu and Wenyin Fu and Brian Fuller and Cynthia Gao and Vedanuj Goswami and Naman Goyal and Anthony Hartshorn and Saghar Hosseini and Rui Hou and Hakan Inan and Marcin Kardas and Viktor Kerkez and Madian Khabsa and Isabel Kloumann and Artem Korenev and Punit Singh Koura and Marie-Anne Lachaux and Thibaut Lavril and Jenya Lee and Diana Liskovich and Yinghai Lu and Yuning Mao and Xavier Martinet and Todor Mihaylov and Pushkar Mishra and Igor Molybog and Yixin Nie and Andrew Poulton and Jeremy Reizenstein and Rashi Rungta and Kalyan Saladi and Alan Schelten and Ruan Silva and Eric Michael Smith and Ranjan Subramanian and Xiaoqing Ellen Tan and Binh Tang and Ross Taylor and Adina Williams and Jian Xiang Kuan and Puxin Xu and Zheng Yan and Iliyan Zarov and Yuchen Zhang and Angela Fan and Melanie Kambadur and Sharan Narang and Aurelien Rodriguez and Robert Stojnic and Sergey Edunov and Thomas Scialom},
    year={2023},
    eprint={2307.09288},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
```bibtex
@article{dao2023flashattention2,
    title={Flash{A}ttention-2: Faster Attention with Better Parallelism and Work Partitioning},
    author={Dao, Tri},
    year={2023}
}
```
```bibtex
@article{jain2023neftune,
    title={NEFTune: Noisy Embeddings Improve Instruction Finetuning},
    author={Jain, Neel and Chiang, Ping-yeh and Wen, Yuxin and Kirchenbauer, John and Chu, Hong-Min and Somepalli, Gowthami and Bartoldson, Brian R and Kailkhura, Bhavya and Schwarzschild, Avi and Saha, Aniruddha and others},
    journal={arXiv preprint arXiv:2310.05914},
    year={2023}
}
```
