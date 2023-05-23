# Evaluation

In this directory, we introduce how you can evaluate your model with our pipeline. This pipeline is available for model
evaluation of Chinese capability and the one for English capability is under preparation.

## Installation
To start model evaluation, you need to install required packages which listed in `requirements.txt` under `evaluate` folder.
```shell
pip install -r requirements.txt
```

## Evaluation Pipeline

The whole evaluation pipeline consists of two methods:
1. `GPT Evaluation`: evaluates model predictions using the GPT-3.5.
   * Compare the performance of two different models (battle).
   * Rate model according to pre-defined metrics using prompting design.
2. `Automatic Evaluation`: evaluates model predictions using automatic metrics.

### Evaluation Category
The model capability is seperated into 10 evaluation categories, which refers to the user case mentioned in InstructGPT.
Following table introduces each category:

| Evaluation Category | Description                                                                                                                              |
|:-------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------|
|      Roleplay       | Given certain characteristic, the capability of chatting as the character                                                                |
|        Chat         | Conduct multiple rounds of dialogue, the capability of understanding and memorization of previous rounds of dialogue                     |
|       Open QA       | Given an open question, the capability of answering questions in opened-ended way                                                        |
|      Closed QA      | Given a closed question, the capability of answering questions with limited scope (such as single/multiple choice question)              |
|    Brainstorming    | Given a question requiring divergent answers, the capability of divergent answering and listing in points                                |
|     Generation      | Given generation task, the capability of generating in high quality and human-written way (such as writing an email)                     |
|      Rewriting      | Given rewriting task, the capability of rewriting sentences to meet task requirements (such as active and passive switches, translation) |
|   Classification    | Given classification task, the capability of accurate classification                                                                     |
|     Extraction      | Given extraction task, the capability of extracting required information                                                                 |
|    Summarization    | Given a paragraph or passage, the capability of summarization                                                                            |

To better understand each evaluation category, here are some prompt examples provided.


| Evaluation Category | Chinese Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | English Example                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|:-------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|      Roleplay       | **Example 1：**<br/>我想让你担任Android开发工程师面试官。我将成为候选人，您将向我询问Android开发工程师职位的面试问题。我希望你只作为面试官回答。不要一次写出所有的问题。我希望你只对我进行采访。问我问题，等待我的回答。不要写解释。像面试官一样一个一个问我，等我回答。我的第一句话是“面试官你好”。 <br/><br/>**Example 2：**<br/>我想让你扮演讲故事的角色。你会想出引人入胜、富有想象力和吸引观众的有趣故事。它可以是童话故事、教育故事或任何其他类型的有潜力的故事以吸引人们的注意力和想象力。根据目标受众，您可以为您的讲故事环节选择特定的主题或主题，例如，如果是儿童，那么您可以谈论动物；如果是成人，那么基于历史的故事可能会更好地吸引他们等。我的第一个请求是我需要一个关于毅力的有趣故事。                                                                                                  | **Example 1：**<br/>I want you to act as a journalist. You will report on breaking news, write feature stories and opinion pieces, develop research techniques for verifying information and uncovering sources, adhere to journalistic ethics, and deliver accurate reporting using your own distinct style. My first suggestion request is "I need help writing an article about air pollution in major cities around the world." <br/><br/>**Example 2:**<br/> I want you to act as a debater. I will provide you with some topics related to current events and your task is to research both sides of the debates, present valid arguments for each side, refute opposing points of view, and draw persuasive conclusions based on evidence. Your goal is to help people come away from discussion with increased knowledge and insight into the topic at hand. My first request is "I want an opinion piece about Deno." |
|        Chat         | **Example 1：**<br/>基于以下角色信息完成一段对话。小王，20岁，是一名大学生；小张，22岁，是一名大学生。<br/> 小王：小张，我们一起去看电影吧？ 小张：好啊，你想看什么电影呢？ 小王：我听说最近有一部很好看的科幻电影。你觉得怎么样？<br/>小张：<br/><br/>**Example 2:**<br/>基于以下角色信息完成一段对话。小明是一名医生，一位老年病患者想要停药，但他对病情有所忽视并有担忧；王叔叔是老年病患者的儿子，希望能够听取医生的建议。<br/>小明：你好，王叔叔，我了解你想要让你父亲停药。<br/>王叔叔：是的，我父亲已经吃了那么久的药，我担心药物对他的身体会有副作用。<br/>小明：                                                                                                                                                        | **Example 1：**<br/>Human: Who is the highest try scorer in the current Sydney Roosters team?<br/>AI: As an AI language model, I don't have access to current statistics. However, you can check the current Sydney Roosters team's official website or relevant sports news websites to find the answer.<br/>Human: Thanks for the suggestion, but can you recommend a reliable sports news website that I can check?<br/>AI:<br/><br/>**Example 2:**<br/>Complete a dialogue based on the following role information. <br/>A: Elementary student  B: Teacher<br/>B: Good morning, Student A. Today we're going to learn about addition and subtraction.<br/>A: Teacher, I already know this very well. Why do I need to learn it again?<br/>B:                                                                                                                                                                               |
|       Open QA       | **Example 1：**<br/>请问万有引力定律由谁提出的？<br/><br/>**Example 2：**<br/>哪些国家参与了第一次世界大战？                                                                                                                                                                                                                                                                                                                                                                                                       | **Example 1：**<br/>Who are the indigenous people of New Zealand?<br/><br/>**Example 2：**<br/>How do you take the derivative of the sin function?                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|      Closed QA      | **Example 1：**<br/>请从以下选项中选择正确答案。以下哪个是世界上最高山峰？ <br/>A. 长城 <br/>B. 泰山 <br/>C. 珠穆朗玛峰 <br/>D. 黄山<br/><br/>**Example 2：**<br/>请从以下选项中选择一个最佳答案回答下面的问题。问题：非洲最高的山是哪座山？<br/> 选项： <br/>A. 麦金利山 <br/>B. 喜马拉雅山 <br/>C. 乞力马扎罗山                                                                                                                                                                                                                                                                  | **Example 1：**<br/>Answer the following question:<br/>What shape is the Earth?<br/>A) A circle<br/>B) A sphere<br/>C) An ellipse<br/>D) A plane<br/><br/>**Example 2：**<br/>Choose the correct classification for the following question:<br/>"What type of restaurant is 'Burger King'"?<br/>fast food<br/>family style<br/>formal dining<br/>buffet<br/>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|    Brainstorming    | **Example 1：**<br/>请介绍一下人工智能的多个领域。<br/><br/>**Example 2：**<br/>请给出管理家庭财务的3个小技巧。<br/>                                                                                                                                                                                                                                                                                                                                                                                                | **Example 1：**<br/>What are 10 science fiction books I should read next?<br/><br/>**Example 2：**<br/>List five ideas for how to regain enthusiasm for my career.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|     Generation      | **Example 1：**<br/>请撰写一篇文章，介绍如何通过改善生活习惯来预防疾病和延长寿命。<br/><br/>**Example 2：**<br/>请根据以下情节撰写一篇短篇小说：一名年轻人被困在一个荒岛上，他必须想办法生存下去直到被救援。但他很快发现自己并不孤单。                                                                                                                                                                                                                                                                                                                                          | **Example 1：**<br/>Can you help me write a formal email to a potential business partner proposing a joint venture?<br/><br/>**Example 2：**<br/>Please use the appropriate format to write a formal letter of recommendation for a student applying to a prestigious computer science graduate program at a university.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|      Rewriting      | **Example 1：**<br/>将以下句子改为被动语态：<br/>"他们正在洗车"<br/><br/>**Example 2：**<br/>将以下文本翻译成英语：<br/>“这个周末我要去海边玩”                                                                                                                                                                                                                                                                                                                                                                               | **Example 1：**<br/>Translate the following text into English: <br/>"我最喜欢的季节是春天，因为我可以看到美丽的花朵。"<br/><br/>**Example 2：**<br/>Please correct the following sentences and give them the correct sentence.<br/>"Their going to the party there."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|   Classification    | **Example 1：**<br/>新闻标题：今日立夏，有一上联，立夏万物并秀，下联怎么对？<br/>请根据以上新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。<br/><br/> **Example 2：**<br/>新闻标题：赵丽颖很久没有登上微博热搜了，但你们别急，她只是在憋大招而已。<br/>请根据新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。                                                                                                                                                                                                                             | **Example 1：**<br/>Classify the given email as spam or non-spam.<br/>"Hello, this is an email reminding you to pay your property fees"<br/><br/>**Example 2：**<br/>Classify the following text as news, ads or forum posts<br/>"The latest iPhone 13 is now available, shop now!"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
|     Extraction      | **Example 1：**<br/>根据以下新闻文本，提取新闻报道时间，例如回答时按照格式“新闻报道时间：2007年8月10日”<br/>新闻文本如下：2007-4-7中新网4月7日电据中国消防在线消息，4月4日晚上7时30分左右，湖南长潭高速公路上发生一起6车连环相撞失火事故。长株潭三地消防部门共出动消防车21台，警力100余人。经过消防官兵近2个小时奋力扑救，大火被成功扑灭。据初步调查，有1人在此次事故中死亡。<br/><br/>**Example 2：**<br/>根据以下新闻文本，提取新闻报道时间，例如回答时按照格式“新闻报道时间：2007年8月10日”<br/>新闻文本如下：2014年1月15日，据外媒《俄罗斯报》报道称，位于北半球的澳大利亚现在正处于炎热的夏季，而近日也到了高温酷暑的时候，当地时间1月14日晚，澳大利亚南部一夜间发生至少250起火灾。受炎热天气及雷雨天气影响，澳大利亚南部一夜间发生至少250起火灾，灾情多集中在维多利亚州。火灾发生后，救援人员立即展开救灾行动。目前，大部分起火点火势已被控制。 | **Example 1：**<br/>Extract all phenotypes of the following text:<br/>"The 55-year-old patient has fever and hypertension."<br/><br/>**Example 2：**<br/>Extract the location mentioned in the following text:<br/>"The student graduated from Harvard university, which is located in Boston"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|    Summarization    | **Example 1：**<br/>请简要总结概括以下段落材料。<br/>新华社快讯：斯里兰卡政府部门21日说，首都科伦坡包括教堂、酒店等多个地点当天发生的爆炸已导致至少70人死亡，另有260多人受伤。<br/><br/> **Example 2：**<br/>请简要总结概括以下段落材料。<br/>近期，参与京雄高铁站站房建设的中铁十二局，因在施工过程中存在环境违法行为被雄安新区公开通报。通报发出后，引起社会广泛关注。近日，人民网记者从雄安新区相关部门及中铁十二局获悉，新区有关部门已经集中约谈了中铁十二局等24个参与雄安建设的项目单位。对于约谈内容和结果，中铁十二局有关宣传负责人回应：“具体内容不清楚，最好找雄安新区相关部门了解情况。”新区有关部门负责人表示，此前涉及的环境违法行为，中铁十二局已基本整改到位，但约谈内容和结果暂不公开，接下来，将按部就班推进环境治理工作。（原题为《雄安新区：中铁十二局涉环境违法已基本整改到位》）                                                | **Example 1：**<br/>Please provide a summary based on the following news：<br/>"China plans to launch its first space station core module in 2022, an important development in the country's space program. The space station, called Tianhe, will include three modules: a core module, an experiment module and an astronomy module. The first launch of the core module will be used to test and verify the basic functions of the station, as well as to conduct related scientific research and technology experiments. "<br/><br/>**Example 2：**<br/>What information is provided in the table below? Summarize the core information in it？<br/>"Ranking, Player Name, Team, Position, Salary (in millions of dollars)<br/>1, LeBron James, Los Angeles Lakers, SF, 45.0<br/>2, Stephen Curry, Golden State Warriors, PG, 43.5"                                                                                           |


### Evaluation Metrics
#### GPT Evaluation
Use GPT-3.5 to evaluate the prediction of different models, and pre-define evaluation metrics for different categories. There are 10 pre-defined evaluation metrics and you can refer to the table below:

|    Evaluation Metric    | Prompt Words                                                 | CoT                                                                                                                                                                                                                                                                       |
|:-----------------------:|:-------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Language organization   | 语言组织(1-5)：答案语言是否流畅、连贯，使用正确的语法，具有一定逻辑性，使用恰当的连接词、过渡词等等。        | 1. 阅读答案，并检查是否有语法错误、用词不当或其他显著的错误。<br/> 2.检查答案是否具有逻辑性，能够按照合理的顺序传达信息并且能够自圆其说<br/> 3. 确定答案是否与问题或主题相关，并且能够传达清晰的信息。<br/> 4. 检查答案是否连贯，是否使用适当的转换和过渡来保持句子和段落之间的连贯性。<br/> 5. 检查答案是否具有明确的结构和组织方式，使得读者可以轻松理解信息的层次和结构。<br/> 6. 根据以上因素综合评估答案的语言组织，并给出一个1到5的分数，其中5表示语言组织非常好，而1表示语言组织非常差。 |
|        Relevance        | 切题(1-5)：答案内容是否切题，不答非所问，并且严格遵照题目要求。                           | 1. 阅读题目，确定题目所问的问题是什么，以及需要回答哪些方面的问题。<br/> 2. 阅读答案，确认答案是否直接回答了题目所问的问题。<br/> 3. 检查答案是否严格遵照了题目的要求，包括答题方式、答题长度、答题格式等等。<br/> 4. 根据以上因素综合评估答案的切题程度，并给出一个1到5的分数，其中5表示答案非常切题，而1表示答案完全没有切题。                                                                                         |
|       Creativity        | 创意性(1-5)：某些头脑风暴问题可能需要答案具有创意，提出新的思路。                          | 1. 仔细阅读所提供的头脑风暴问题，确保你理解问题的要点和背景。<br/> 2. 根据你的知识和经验，判断所提供的答案是否可行。如果答案不可行，则创意性评分可能会受到影响。<br/> 3. 考虑答案中是否包含新颖的想法或独特的思路。答案可能与已知的解决方案有所重叠，但仍然可以被认为是有创意的，只要它提供了新的角度或方法来解决问题。<br/> 4. 根据答案的创意性，给出一个1到5的评分。如果答案缺乏创意，则应给出一个较低的评分。如果答案具有创意并提供了新的思路，应给出一个较高的评分。                      |
|      Practicality       | 实用性(1-5)：某些头脑风暴问题可能需要答案提出实用的建议或解决方法。                         | 1. 仔细阅读所提供的头脑风暴问题，确保你理解问题的要点和背景。<br/> 2. 根据你的知识和经验，判断所提供的答案是否可行。如果答案不可行，则实用性评分可能会受到影响。<br/> 3. 考虑答案中提出的建议或解决方法是否实用并可行。答案可能看起来很好，但如果无法实现或应用，则实用性评分可能会受到影响。<br/> 4. 根据答案的实用性，给出一个1到5的评分。如果答案缺乏实用性，则应给出一个较低的评分。如果答案提出了实用的建议或解决方法，并且可以很好地解决问题，则应给出一个较高的评分。                    |
|       Correctness       | 正确性(1-5)：答案应该符合常识、生活实际等等                                     | 1. 仔细阅读所提供的头脑风暴问题，确保你理解问题的要点和背景。<br/> 2. 根据你的知识和经验，判断所提供的答案是否可行。如果答案不可行，则正确性评分可能会受到影响。<br/> 3. 考虑答案中所提供的信息是否正确、符合常识、生活实际等等。如果答案中存在明显的错误或不合理之处，则正确性评分可能会受到影响。<br/> 4. 根据答案的正确性，给出一个1到5的评分。如果答案存在明显的错误或不合理之处，则应给出一个较低的评分。如果答案正确、符合常识、生活实际等等，则应给出一个较高的评分。                    |
|       Naturalness       | 自然(1-5)：答案是否自然，并且符合问题给定的身份。                                  | 1. 阅读题目，确定题目提供的身份信息。<br/> 2. 检查答案内容是否符合题目给定的身份。<br/> 3. 根据以上因素，对该回答的自然性进行打分，分数从1到5，其中1表示不自然，5表示非常自然，并符合问题给定的身份。                                                                                                                                                           |
|      Engagingness       | 参与感(1-5)：答案是否对前面的对话内容做出了恰当的反应，是否理解对话的语境和背景。                  | 1. 阅读题目，确定对话的语境和背景。<br/> 2. 检查答案是否充分理解对话的语境和背景，能否自然地融入到对话中而不显得突兀。<br/> 3. 根据以上因素，对该回答的参与感进行打分，分数从1到5，其中1表示没有参与感，5表示非常有参与感，并且恰当地理解了对话的语境和背景。                                                                                                                               |
|     Reasonableness      | 合理性(1-5)：答案是否能够与前面的对话内容形成逻辑上的衔接，是否符合常理，能否在这个上下文中合理存在。        | 1. 阅读题目，确定对话的主题以及问题期望的回答方向。<br/> 2. 判断答案是否能够与前面的对话内容形成逻辑上的衔接，是否符合常理，能否在这个上下文中合理存在。<br/> 3. 根据以上因素，对该回答的合理性进行打分，分数从1到5，其中1表示不合理，5表示非常合理，并且能够与前面的对话内容形成逻辑上的衔接，并符合常理。                                                                                                        |
|        Diversity        | 多样性(1-5)：答案使用语言是否优美，具有有一定的创造性和想象力。然而，回答也应该保持合理和适度，不要过于夸张或离题。 | 1. 仔细阅读整个回答，确保完全理解回答所表达的内容和主题。<br/> 2. 在阅读回答的同时，注意语言的质量，例如措辞是否正确，语言是否生动等。<br/> 3. 检查回答的创造性和想象力，看看回答是否能够吸引人阅读下去。<br/> 4. 检查回答的合理性和适度，看看回答是否夸张或离题。5. 将多样性的评分打分在1到5之间，5分表示回答的质量很好，能够吸引人阅读，1分表示回答的内容生硬或者有离题的问题。                                                               |
|        Fidelity         | 保真度(1-5)：答案是否能够严格遵守角色的设定回答给定的请求。                             | 1. 仔细阅读问题，了解角色在问题中的设定和表现，包括职业、背景、观点、性格等方面。<br/> 阅读题目的请求，确认回答请求时需要注意的细节。<br/> 3. 对比提供的回答与该角色的设定，评估回答是否能够严格遵守角色的设定。<br/> 4. 结合以上评估结果给出保真度的评分，范围从1到5分，其中1分表示回答与角色设定完全不符，5分表示回答完全符合角色设定且满足给定请求。                                                                               |
|       Conciseness       | 简明扼要(1-5)：答案是否简明扼要，没有冗余内容。                                   | 1. 阅读题目，提取出材料的重点。<br/> 2. 阅读该总结，并注意其中的主要观点和信息。<br/> 3. 评估总结的长度。一个简明扼要的总结通常应该在几句话或几段文字内传达关键信息，而不是冗长的段落或文章。<br/> 4. 检查总结是否包含与主要观点无关的信息或冗余信息。<br/> 5. 确定总结涵盖了材料中的关键信息，并且没有忽略任何重要细节。<br/> 6. 给总结打出1-5的分数，其中5表示总结简明扼要，没有冗余内容，而1表示总结冗长或包含不必要的信息，难以理解或记忆。根据您的判断，打出适当的得分。         |

GPT-3.5 evaluates the quality of model predictions based on the given prompt words and gives a score between 1-5.

#### Automatic Evaluation
Automated metrics evaluate the capability of a model by comparing model predictions with reference answers.
There are two ways to obtain reference answers:
* For instruction coming from human-designed problems, the reference answers are generated by GPT-3.5, such as roleplay, chat.
* For instruction related with classic NLP problems, the reference answers are collected from open-sourced dataset with target answers, such as classification, extraction, summarization.

There are 5 types of automatic evaluation metrics listed in the table below:

 |     Automatic Evaluation Metric     | Description                                                                                                                                                                                        |
|:-----------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|               BLEU-n                | Measure the accuracy between prediction and reference.<br/> BLEU-1 (Unigram) evaluates accuracy in word level<br/> BLEU-n (n-gram) evaluate the fluency in sentence level.                         |
|                ROUGE                | ROUGE-N measures the number of matching n-grams between prediction and reference. <br/> ROUGE-L measures the number of matching longest common subsequence (LCS) between prediction and reference. |
|              Distinct               | Measure the diversity of generation text by counting the unique n-grams.                                                                                                                           |
|              BERTScore              | Measure the semantic similarity between tokens of predictions and references with BERT.                                                                                                            |
| Precision<br/> Recall<br/> F1 Score | Measure the number of overlaps between prediction and reference (design for classification and extraction categories)                                                                              |

## Evaluation Process
### Data Format
#### Target Answers / Predictions
A JSON file contains one list. Each element in the list is a target answer / prediction record for one instruction / question.
An element should have the following fields:

* `category` (str, compulsory): The category of the instruction / question.
* `instruction` (str, compulsory): The instruction / question for the LLM.
* `input` (str, optional): The additional context of the instruction / question.
* `output` (str, optional): The sample output of the instruction (default: GPT-3.5).
* `target` (str, optional): The target answer for the instruction.
* `id` (int, compulsory): The ID of the instruction / question.

If the `input` has a target answer, the `output` can be empty. Otherwise, we generate answers from GPT-3.5 as the `output`, and the `target` field is empty.

Example:
```
[
    {
        "category": "brainstorming",
        "instruction": "请介绍一下人工智能的多个领域。",
        "input": "",
        "output": "{GPT-3.5 Answers}",
        "target": "",
        "id": 1
    },
    {
        "category": "classification",
        "instruction": "新闻标题：为什么电影《倩女幽魂》中燕赤霞一个道士却拿着金刚经？请根据新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。",
        "input": "",
        "output": "",
        "target": "{target answer}",
        "id": 2
    }
]
```

#### Model Answers / Predictions

A JSON file contains one list. Each element in the list is a model answer / prediction record for one instruction / question.

An element should have the following fields:

* `category` (str, compulsory): The category of the instruction / question.
* `instruction` (str, compulsory): The instruction / question for the LLM.
* `input` (str, optional): The additional context of the instruction / question.
* `output` (str, compulsory): The output from the LLM.
* `target` (str, optional): The target answer for the instruction.
* `id` (int, compulsory): The ID of the instruction / question.

Example:
```
[
    {
        "category": "brainstorming",
        "instruction": "请介绍一下人工智能的多个领域。",
        "input": "",
        "output": "{Model Answers / Predictions}",
        "target": "",
        "id": 1
    },
    {
        "category": "classification",
        "instruction": "新闻标题：为什么电影《倩女幽魂》中燕赤霞一个道士却拿着金刚经？请根据新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。",
        "input": "",
        "output": "{Model Answers / Predictions}",
        "target": "{target answer}",
        "id": 2
    }
]
```

### Evaluation
#### Configuration
The configuration file `config_cn.json` can control how evaluate the performance of the model.
The following is an example showing the config structure:
```
{
    "language": "cn",
    "category": {
        "brainstorming": {
            "GPT-3.5": ["relevance", "creativity", "practicality", "correctness"],
            "Metrics": ["Distinct"]
        },
        "chat": {
            "GPT-3.5": [ "relevance", "naturalness", "engagingness", "reasonableness"],
            "Metrics": ["Distinct"]
        }
    }
}
```
`"language"`: evaluate the model capability in which language, we only support Chinese `"cn"` for now.
`"category"`: evaluate the model capability in which category/categories.
`"GPT-3.5"`: config metrics for GPT-3.5 evaluation.
`"Metrics"`: config metrics for automatic metrics evaluation.

You can create your config file based on available settings listed in following table.

|    "category"    |        "GPT-3.5"        |  "Metrics"  |
|:----------------:|:-----------------------:|:-----------:|
| "brainstorming"  | "language organization" |   "BLEU"    |
|      "chat"      |       "relevance"       |   "ROUGE"   |
| "classification" |      "creativity"       | "Distinct"  |
|   "closed_qa"    |     "practicality"      | "BERTScore" |
|   "extraction"   |      "correctness"      | "Precision" |
|   "generation"   |      "naturalness"      |  "Recall"   |
|    "open_qa"     |     "engagingness"      | "F1 score"  |
|   "rewriting"    |    "reasonableness"     |
|    "roleplay"    |       "diversity"       |
| "summarization"  |       "fidelity"        |
|                  |      "conciseness"      |

#### Evaluate
After setting the configuration file, you can evaluate the model using `eval.py`.


An example script is provided as follows:
```shell
python eval.py \
    --config_file "path to the config file" \
    --battle_prompt_file "path to the prompt file for battle" \
    --gpt_evaluation_prompt_file "path to the prompt file for gpt evaluation" \
    --target_file "path to the target answer file" \
    --answer_file_list "path to the answer files of at most 2 models" \
    --model_name_list "the names of at most 2 models" \
    --save_path "path to save results" \
    --openai_key "your openai key" \
```

## To Do
- [ ] Add evaluation for English capability
- [ ] Support UniEval
- [ ] Support GPT-4 evaluation

## Citations

```bibtex
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://vicuna.lmsys.org},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}

@misc{ouyang2022training,
      title={Training language models to follow instructions with human feedback},
      author={Long Ouyang and Jeff Wu and Xu Jiang and Diogo Almeida and Carroll L. Wainwright and Pamela Mishkin and Chong Zhang and Sandhini Agarwal and Katarina Slama and Alex Ray and John Schulman and Jacob Hilton and Fraser Kelton and Luke Miller and Maddie Simens and Amanda Askell and Peter Welinder and Paul Christiano and Jan Leike and Ryan Lowe},
      year={2022},
      eprint={2203.02155},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
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
