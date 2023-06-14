# Evaluation

In this directory, we introduce how you can evaluate your model with our pipeline. This pipeline is now available for evaluation of both Chinese and English capability.

## Installation

To start model evaluation, you need to install required packages which listed in `requirements.txt` under `evaluate` folder.

```shell
pip install -r requirements.txt
```

## Evaluation Pipeline

The whole evaluation pipeline consists of three methods:

1. `GPT Evaluation`: evaluates model predictions using GPT models.
   * Compare the performance of two different models (battle).
   * Rate the model according to pre-defined metrics using prompting design.
   * Rate the model according to pre-defined metrics with additional reference answer using prompting design.
2. `Automatic Evaluation`: evaluates model predictions using automatic metrics.
3. `UniEval`: evaluates model predictions using UniEval models(English only).

### Evaluation Category

Our evaluation pipeline examines the model's capability using 10 categories of questions. The following table introduces each category:

| Evaluation Category | Description                                                  |
| :-----------------: | :----------------------------------------------------------- |
|    Brainstorming    | Models are asked to generate a range of creative and diverse ideas according to the question. The capability of creativity is required. |
|        Chat         | Models are asked to continue a multi-round dialogue given the roles involved. The capability of understanding, memorizing previous rounds of the dialogue and answering according to the persona provided is required. |
|   Classification    | Models are asked to do classification tasks. The capability of accurate classification is required. |
|      Closed QA      | Models are asked to answer a closed QA question. The capability of answering questions with limited scope (such as single/multiple choice question) is required. |
|     Extraction      | Models are asked to extract information from a given material. The capability of extracting required information is required. |
|     Generation      | Models are asked to generate an email, letter, article, etc. The capability of generating texts in a high quality and human-written way is required. |
|       Open QA       | Models are asked to answer an open QA question(without context provided). The capability of answering questions with the models' own knowledge base is required. |
|      Roleplay       | Models are asked to play the role provided. The capability of engaging in the scenario and effectively interacting with the user is required. |
|      Rewriting      | Models are asked to do rewriting tasks such as translation and grammar correction. The capability of rewriting according to different instructions is required. |
|    Summarization    | Models are asked to summarize the given paragraph or passage. The capability of summarization is required. |

To better understand each evaluation category, here are some example questions provided.


| Evaluation Category | Chinese Example                                              | English Example                                              |
| :-----------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|    Brainstorming    | **Example 1:**<br/>请介绍一下人工智能的多个领域。<br/><br/>**Example 2:**<br/>请给出管理家庭财务的3个小技巧。<br/> | **Example 1:**<br/>How can I improve my memory? Any useful techniques you can suggest?<br/><br/>**Example 2:**<br/>What are some ways to increase productivity while working from home? |
|        Chat         | **Example 1:**<br/>基于以下角色信息完成一段对话。小张是一名新手爱好者，对养鸡有浓厚的兴趣。老李是一名有丰富经验的养鸡大师。<br/>小张：您好，老李，我最近开始对养鸡感兴趣了，想请教您一些问题。 <br/>老李：你好，小张，我很乐意帮助你。你想问些什么？ <br/>小张：我想知道如何确定鸡的品种和性别？ <br/>老李：确切的品种可以通过鸡的外貌特征来确定，而性别一般是通过鸡卵的大小和形状来判断。还有什么问题吗？<br/> 小张：<br/><br/>**Example 2:**<br/>基于以下角色信息完成一段对话。小明是一名医生，一位老年病患者想要停药，但他对病情有所忽视并有担忧；王叔叔是老年病患者的儿子，希望能够听取医生的建议。<br/>小明：你好，王叔叔，我了解你想要让你父亲停药。<br/>王叔叔：是的，我父亲已经吃了那么久的药，我担心药物对他的身体会有副作用。<br/>小明： | **Example 1:**<br/>Complete a conversation based on the following character information. Amy is a 30-year-old chef who runs her own restaurant. Jack is a food blogger who specializes in reviewing local restaurants.<br/>Amy: Hi Jack, I heard that you're a food blogger. Nice to meet you. <br/>Jack: Hi Amy, yes I am. Your restaurant has been receiving a lot of good reviews lately. <br/>Amy: Yes, we use only fresh and quality ingredients, and every dish is carefully crafted. <br/>Jack: <br/><br/>**Example 2:**<br/>Complete a dialogue based on the following role information. A: Elementary student  B: Teacher<br/>B: Good morning, Student A. Today we're going to learn about addition and subtraction.<br/>A: Teacher, I already know this very well. Why do I need to learn it again?<br/>B: |
|   Classification    | **Example 1:**<br/>新闻标题：今日立夏，有一上联，立夏万物并秀，下联怎么对？<br/>请根据以上新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。<br/><br/> **Example 2:**<br/>新闻标题：赵丽颖很久没有登上微博热搜了，但你们别急，她只是在憋大招而已。<br/>请根据新闻标题判断新闻所属的分类，你需要从文化，娱乐，体育，财经，房产，教育，科技，旅游，游戏，军事这十类中选择一个答案。 | **Example 1:**<br/>Title:  Fighting for Love (2020) <br/>Description: Jasmine got obsessed with a man and now he's obsessed with her. Steamy nights, kisses and rules being broken awaits them. She turned his whole world upside down and now he's doing it to hers. In this free fall, can they survive each others love?\"<br/>Based on the above information, determine which genre the work of art belongs to. You can only choose one from \"sport\", \"horror\", \"drama\", \"history\", \"romance\", \"biography\", \"science fiction\", \"comedy\", \"animation\", \"documentary\", \"music\" and \"news\".<br/><br/>**Example2:** <br/>Title:  Summer Breeze: The Isley Brothers Greatest Hits Live (2005)<br/>Description: Filmed in the US in 2005 and captured in excellent form led by Ron Isley's vocals and Ernie Isley's hard edged guitar. Virtually every track is a hit including Shout, Who's That Lady, Twist And Shout, Summer Breeze and Harvest For The World.<br/>Based on the above information, determine which genre the work of art belongs to. You can only choose one from \"sport\", \"horror\", \"drama\", \"history\", \"romance\", \"biography\", \"science fiction\", \"comedy\", \"animation\", \"documentary\", \"music\" and \"news\"." |
|      Closed QA      | **Example 1:**<br/>请从以下选项中选择正确答案。以下哪个是世界上最高山峰？ <br/>A. 长城 <br/>B. 泰山 <br/>C. 珠穆朗玛峰 <br/>D. 黄山<br/><br/>**Example 2:**<br/>请从以下选项中选择一个最佳答案回答下面的问题。问题：非洲最高的山是哪座山？<br/> 选项： <br/>A. 麦金利山 <br/>B. 喜马拉雅山 <br/>C. 乞力马扎罗山 | **Example 1:**<br/>Which of the following options is NOT a primary color?<br/>(a) yellow<br/>(b) blue<br/>(c) orange<br/>(d) red<br/><br/>**Example 2:**<br/>Choose the correct option to complete the following sentence: \"Harry Potter and the Chamber of Secrets\" is the ________ book in the Harry Potter series.<br/>(A) first<br/>(B) second<br/>(C) third<br/>(D) fourth |
|     Extraction      | **Example 1:**<br/>根据以下新闻文本，提取新闻报道时间，例如回答时按照格式“新闻报道时间：2007年8月10日”<br/>新闻文本如下：2007-4-7中新网4月7日电据中国消防在线消息，4月4日晚上7时30分左右，湖南长潭高速公路上发生一起6车连环相撞失火事故。长株潭三地消防部门共出动消防车21台，警力100余人。经过消防官兵近2个小时奋力扑救，大火被成功扑灭。据初步调查，有1人在此次事故中死亡。<br/><br/>**Example 2:**<br/>根据以下新闻文本，提取新闻报道时间，例如回答时按照格式“新闻报道时间：2007年8月10日”<br/>新闻文本如下：2014年1月15日，据外媒《俄罗斯报》报道称，位于北半球的澳大利亚现在正处于炎热的夏季，而近日也到了高温酷暑的时候，当地时间1月14日晚，澳大利亚南部一夜间发生至少250起火灾。受炎热天气及雷雨天气影响，澳大利亚南部一夜间发生至少250起火灾，灾情多集中在维多利亚州。火灾发生后，救援人员立即展开救灾行动。目前，大部分起火点火势已被控制。 | **Example 1:**<br/>Ernest Hemingway, an American literary giant known for his spare and direct writing style, has penned timeless works such as 'The Old Man and the Sea', 'For Whom the Bell Tolls', and 'A Farewell to Arms', which have made a profound impact on the literary world and continue to be widely read and admired today.<br/>Extract the name of the author mentioned above.<br/><br/>**Example 2:**<br/>In the epic fantasy series 'A Song of Ice and Fire', George R.R. Martin weaves a complex web of political intrigue, war, and magic across the fictional continents of Westeros and Essos. Martin's richly developed characters and intricate plotlines have captivated readers worldwide, much like his other acclaimed works such as 'A Clash of Kings' and 'A Storm of Swords'.<br/>Extract the name of the author in the above material. |
|     Generation      | **Example 1:**<br/>请撰写一篇文章，介绍如何通过改善生活习惯来预防疾病和延长寿命。<br/><br/>**Example 2:**<br/>请根据以下情节撰写一篇短篇小说：一名年轻人被困在一个荒岛上，他必须想办法生存下去直到被救援。但他很快发现自己并不孤单。 | **Example 1:**<br/>Write a descriptive paragraph about an island to relax and unwind, including details about the location and atmosphere.<br/><br/>**Example 2:**<br/>Can you help me write a persuasive email to my colleagues encouraging them to participate in a charitable fundraising event? |
|       Open QA       | **Example 1:**<br/>请问万有引力定律由谁提出的？<br/><br/>**Example 2:**<br/>哪些国家参与了第一次世界大战？ | **Example 1:**<br/>What are the four basic tastes of the human palate?<br/><br/>**Example 2:**<br/>Who painted the The Scream? |
|      Rewriting      | **Example 1:**<br/>请将以下句子改为正确的语序。 <br/>生日快乐你祝他了吗？<br/><br/>**Example 2:**<br/>将以下文本翻译成英语：<br/>“这个周末我要去海边玩” | **Example 1:**<br/>Please translate the following sentences, which are a mixture of Chinese and English, into full English. <br/>我需要买一些healthy snacks，比如nuts和dried fruits，作为我的office的午餐.<br/><br/>**Example 2:**<br/>Please rewrite the sentence using an inverted sentence structure.<br/>We won't begin our journey until the sun sets. |
|      Roleplay       | **Example 1:**<br/>我想让你担任Android开发工程师面试官。我将成为候选人，您将向我询问Android开发工程师职位的面试问题。我希望你只作为面试官回答。不要一次写出所有的问题。我希望你只对我进行采访。问我问题，等待我的回答。不要写解释。像面试官一样一个一个问我，等我回答。我的第一句话是“面试官你好”。 <br/><br/>**Example 2:**<br/>我想让你扮演讲故事的角色。你会想出引人入胜、富有想象力和吸引观众的有趣故事。它可以是童话故事、教育故事或任何其他类型的有潜力的故事以吸引人们的注意力和想象力。根据目标受众，您可以为您的讲故事环节选择特定的主题或主题，例如，如果是儿童，那么您可以谈论动物；如果是成人，那么基于历史的故事可能会更好地吸引他们等。我的第一个请求是我需要一个关于毅力的有趣故事。 | **Example 1:**<br/>Assume the role of a marriage counselor. Develop a series of communication exercises for a couple who are experiencing difficulties in their relationship. These exercises should promote active listening, empathy, and effective expression of emotions. Your first assignment is to provide a set of three exercises that focus on resolving conflicts and rebuilding trust. <br/><br/>**Example 2:**<br/>I want you to act as a travel agent. I will tell you my desired destination, travel dates, and budget, and it will be your job to suggest the best travel itinerary for me. Your recommendations should include the best transportation options, hotel accommodations, and any popular tourist attractions nearby. My first request is "I want to plan a trip to Tokyo for a week, with a budget of $2000. I want to explore the culture and food of the city." |
|    Summarization    | **Example 1:**<br/>请简要总结概括以下段落材料。<br/>当地时间29日，泰国卫生部通报，新增143名新冠肺炎确诊病例和1名死亡病例。截止到当地时间29日上午，泰国累计确诊病例1388例，其中泰国籍1172例，非泰国籍216例。死亡病例累计7例。（原题为《泰国新增143例新冠肺炎确诊病例累计确诊1388例》）<br/><br/> **Example 2:**<br/>请简要总结概括以下段落材料。<br/>近期，参与京雄高铁站站房建设的中铁十二局，因在施工过程中存在环境违法行为被雄安新区公开通报。通报发出后，引起社会广泛关注。近日，人民网记者从雄安新区相关部门及中铁十二局获悉，新区有关部门已经集中约谈了中铁十二局等24个参与雄安建设的项目单位。对于约谈内容和结果，中铁十二局有关宣传负责人回应：“具体内容不清楚，最好找雄安新区相关部门了解情况。”新区有关部门负责人表示，此前涉及的环境违法行为，中铁十二局已基本整改到位，但约谈内容和结果暂不公开，接下来，将按部就班推进环境治理工作。（原题为《雄安新区：中铁十二局涉环境违法已基本整改到位》） | **Example 1:**<br/>The 21 year-old-woman was treated by paramedics after the kitchen fire in Botfield Road in Shifnal, Shropshire. West Mercia Police said it is treating Wednesday morning's incident as arson and are appealing for any witnesses to contact them.The 50-year-old man has been arrested on suspicion of arson with intent to endanger life. For more on this and other stories from Shropshire.<br/>Please briefly summarize the above material within 20 words.<br/><br/>**Example 2:**<br/>South Wales Police were called to a property in Heolgerrig, Merthyr Tydfil, at about 13:40 BST on Sunday. The child was airlifted to Prince Charles Hospital but died shortly afterwards. Police are investigating the circumstances surrounding the incident and have appealed for witnesses. The girl's family are being supported by specially trained officers.<br/>Please briefly summarize the above material within 20 words. |


### Evaluation Metrics

#### GPT Evaluation

GPT evaluation uses GPT models to evaluate the prediction of different models and different pre-defined evaluation metrics are applied to different categories. The following table shows the 11 pre-defined evaluation metrics both in Chinese and English:

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
|      简明扼要<br/>(Conciseness)      | 简明扼要(1-5)：答案是否简明扼要，没有冗余内容。</br></br>Conciseness (1-5): answers should be concise and without redundant content. | 1. 阅读题目，提取出材料的重点。<br/> 2. 阅读该总结，并注意其中的主要观点和信息。<br/> 3. 评估总结的长度。一个简明扼要的总结通常应该在几句话或几段文字内传达关键信息，而不是冗长的段落或文章。<br/> 4. 检查总结是否包含与主要观点无关的信息或冗余信息。<br/> 5. 确定总结涵盖了材料中的关键信息，并且没有忽略任何重要细节。<br/> 6. 给总结打出1-5的分数，其中5表示总结简明扼要，没有冗余内容，而1表示总结冗长或包含不必要的信息，难以理解或记忆。根据您的判断，打出适当的得分。</br></br>1. Read the title and extract the main points of the material.<br>2. Read the summary and note the main ideas and messages in it.<br>3. Assess the length of the summary. A concise summary should usually convey key information within a few sentences or paragraphs, rather than lengthy paragraphs or essays.<br>4. Check that the summary does not contain information that is not relevant to the main ideas or that is redundant.<br>5. Make sure that the summary covers the key information in the material and that no important details have been omitted.<br>6. Rate the summary on a scale of 1-5, where 5 means the summary is concise and free of redundancy, and 1 means the summary is lengthy or contains unnecessary information that is difficult to understand or remember. Based on your judgment, assign the appropriate score. |

GPT models evaluate the quality of model predictions based on the given prompt words and gives a score between 1-5.

> **NOTE 1:**  Even for the same metric, the details of its prompt words and CoT(Chain-of-Thought) can differ based on which category you want to evaluate. For example, prompt words for metric `correctness` showed here is "Whether the answer is correct or not."(this is for category `classification`), but for category `extraction`, prompt words can be "Answers should extract the required information accurately and should not contain any incorrect or misleading information." You can find all the prompt words and CoT(Chain-of-Thought) in `prompt/evaluation_prompt`.

> **NOTE 2:** To add customized metrics, you can refer to [FAQ](#faq).

#### Automatic Evaluation

Automated metrics evaluate the capability of a model by comparing model predictions with reference answers.
There are two ways to obtain reference answers:

* For instruction coming from human-designed problems, the reference answers are generated by GPT-3.5, such as roleplay, chat.
* For instruction related with classic NLP problems, the reference answers are collected from open-sourced dataset with target answers, such as classification, extraction, summarization.

There are 6 types of automatic evaluation metrics listed in the table below:

|     Automatic Evaluation Metric     | Description                                                  |
| :---------------------------------: | :----------------------------------------------------------- |
|               BLEU-n                | Measure the accuracy between prediction and reference.<br/> BLEU-1 (Unigram) evaluates accuracy in word level.<br/> BLEU-n (n-gram) evaluate the fluency in sentence level. |
|                ROUGE                | ROUGE-N measures the number of matching n-grams between prediction and reference. <br/> ROUGE-L measures the number of matching longest common subsequence (LCS) between prediction and reference. |
|              Distinct               | Measure the diversity of generation text by counting the unique n-grams. |
|              BERTScore              | Measure the semantic similarity between tokens of predictions and references with BERT. |
| Precision<br/> Recall<br/> F1 Score | Measure the number of overlaps between prediction and reference (design for classification and extraction categories). |
|                CHRF                 | Measure the similarity of character n-grams between prediction and reference. |

#### UniEval Evaluation

UniEval converts all evaluation tasks of different dimensions(metrics) into Boolean QA problems and utilize the model to answer with “Yes” or “No”. Compared with similarity-based metrics such as ROUGE and BLEU, UniEval can achieve a more comprehensive evaluation. In addition, UniEval also demonstrates its ability to transfer to unseen dimensions and tasks.

In our evaluation pipeline, two pre-trained UniEval evaluators are used. One is [unieval-sum](https://huggingface.co/MingZhong/unieval-sum) and the other is [unieval-dialog](https://huggingface.co/MingZhong/unieval-dialog). The two models can be used for the 3 tasks, `summarization`, `dialogue` and `data2text`. Each task has different evaluation dimensions.

| UniEval Model  | Task               | Dimension(Metric) |
| :------------: | :----------------- | :--- |
| unieval-sum    | summarization | coherence: whether the summary is coherent<br/>consistency: whether the claim is consistent with the given document<br/>fluency: whether the paragraph is fluent<br/>relevance: whether the summary is relevant to the reference |
| unieval-sum | data2text | naturalness: whether the utterance is fluent<br/>informativeness: whether the utterance is informative according to the reference |
| unieval-dialog | dialogue | naturalness: whether the response is natural in the dialogue<br/>coherence: whether the response is coherent in the dialogue history<br/>understandability: whether the response is understandable in the dialogue |

> **NOTE 1:**  Task "data2text" uses the same model as task "summarization".

> **NOTE 2:**  In UniEval paper, the `unieval-sum` model demonstrates the best transfer ability and so you can evaluate your customized metric with this model. Details of adding customized metrics can be found in [FAQ](#faq).

> **NOTE 3:**  We consider not including all metrics provided in UniEval in our pipeline because the data structure and content of the instructions we want to evaluate are not suitable for direct use of some UniEval metrics.

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

```json
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

```json
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

### Prompt

#### Battle Prompt

The following is the Chinese battle prompt. In the battle prompt, the question and answers from two different models are fed into the prompt template. You can find example battle prompt files for Chinese and English in `prompt/battle_prompt`.

```json
{
  "id": 1,
  "system_prompt": "你是一个检查回答质量的好助手。",
  "prompt_template": "[问题]\n{question}\n\n[1号AI助手的答案]\n{answer_1}\n\n[1号AI助手答案终止]\n\n[2号AI助手的答	案]\n{answer_2}\n\n[2号AI助手答案终止]\n\n[要求]\n{prompt}\n\n",
  "prompt": "我们需要你评价这两个AI助手回答的性能。\n请对他们的回答的有用性、相关性、准确性、详细程度进行评分。每个AI助手都会得到一个1到10分的总分，分数越高表示整体表现越好。\n请首先输出一行，该行只包含两个数值，分别表示1号和2号AI助手的分数。这两个分数之间要有一个空格。在随后的一行中，请对你的评价作出全面的解释，避免任何潜在的偏见，并确保AI助手回答的顺序不会影响您的判断。"
}
```

#### Evaluation Prompt

The following is an example of a Chinese GPT evaluation prompt. In an evaluation prompt, you should define your metrics in `metrics` and provide CoT(Chain-of-Thought) in `CoT`.  You can find example evaluation prompt files for Chinese and English in `prompt/evaluation_prompt`.

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

The following is an example of a Chinese config file. The configuration file can control how the pipeline evaluates the model. You need to specify GPT evaluation metrics, automatic metrics and UniEval metrics in key `GPT`, `Metrics` and `UniEval`(English only). You can find an example English config file in `config`.

```json
{
    "language": "en",
    "path_for_UniEval": {
        "summarization": "path to unieval-sum model",
        "dialogue": "path to unieval-dialog model",
        "data2text": "path to unieval-sum model"
    },
    "category": {
        "brainstorming": {
            "GPT": ["relevance", "creativity", "practicality", "reasonableness"],
            "Metrics": ["Distinct"],
            "UniEval": ["summarization-fluency", "data2text-naturalness", "data2text-informativeness"]
        },
        "chat": {
            "GPT": [ "relevance", "naturalness", "engagingness", "reasonableness"],
            "Metrics": ["Distinct"],
            "UniEval": ["dialogue-naturalness", "dialogue-coherence", "dialogue-understandability"]
        }
    }
}
```

`"language"`: the language used to evaluate the model capability. We only support Chinese `"cn"` for now.

`"path_for_UniEval"`: path to the UniEval model.

`"category"`: the category/categories needed to evaluate the model capability.

`"GPT"`: the metrics you want to use for GPT evaluation.

`"Metrics"`: the metrics you want to use for automatic metrics evaluation.

`"UniEval"`: the metrics you want to use for UniEval metrics evaluation. The metric has to be in the `"{task}-{metric}"` format because different tasks have same metrics such as naturalness and coherence.

You can remove the key such as `"Metrics"` to skip evaluating answers using its corresponding evaluation metrics.

You can create your config file based on available settings listed in following table.

|    "category"    |          "GPT"          |  "Metrics"  |          "UniEval"           |
| :--------------: | :---------------------: | :---------: | :--------------------------: |
| "brainstorming"  | "language organization" |   "BLEU"    |    "dialogue-naturalness"    |
|      "chat"      |       "relevance"       |   "ROUGE"   |     "dialogue-coherence"     |
| "classification" |      "creativity"       | "Distinct"  | "dialogue-understandability" |
|   "closed_qa"    |     "practicality"      | "BERTScore" |   "data2text-naturalness"    |
|   "extraction"   |      "correctness"      | "Precision" | "data2text-informativeness"  |
|   "generation"   |      "naturalness"      |  "Recall"   |  "summarization-coherence"   |
|    "open_qa"     |     "engagingness"      | "F1 score"  | "summarization-consistency"  |
|   "rewriting"    |    "reasonableness"     |   "CHRF"    |   "summarization-fluency"    |
|    "roleplay"    |       "diversity"       |             |  "summarization-relevance"   |
| "summarization"  |       "fidelity"        |             |                              |
|                  |      "conciseness"      |             |                              |

> **NOTE:**  For categories which don't have standard answers such as `brainstorming`, you should avoid using automatic metrics such as `BLEU` and `ROUGE` which are based on similarity measures and you should use `Distinct` instead in your config file.

#### Evaluate

After setting the configuration file, you can evaluate the model using `eval.py`. If you want to make comparisons between answers of two different models, you should specify two answer files in the argument `answer_file_list` and two model names in the argument `model_name_list`. If you want to evaluate one answer file, the length of both `answer_file_list` and `model_name_list` should be 1 and the program will perform evaluation using automatic metrics and GPT models.

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

If you want GPT evaluation with reference, you can add an argument `--gpt_with_reference`.

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

<details><summary><b>How can I add a new UniEval evaluation metric?</b></summary>

For example, if you want to add a new metric `persuasiveness` into task `data2text`, you should add a Boolean QA question about the metric in function `add_question` in `unieval/utils.py`. Please do note that how effectively the model would evaluate this metric is unknown and you may need some experiments to test whether the model is capable of evaluating this metric.

```python
if task == 'data2text':
	if dimension == 'persuasiveness':
		cur_input = 'question: Is this a persuasive utterence </s> utterance: ' + output[i]
```

</details>

## To Do

- [x] Add evaluation for English capability
- [x] Support UniEval
- [x] Support GPT-4 evaluation
- [x] Support GPT evaluation with reference

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

@misc{zhong2022unified,
      title={Towards a Unified Multi-Dimensional Evaluator for Text Generation},
      author={Ming Zhong and Yang Liu and Da Yin and Yuning Mao and Yizhu Jiao and Pengfei Liu and Chenguang Zhu and Heng Ji and Jiawei Han},
      year={2022},
      eprint={2210.07197},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
