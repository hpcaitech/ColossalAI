"""
All custom prompt templates are defined here.
"""

from langchain.prompts.prompt import PromptTemplate

# Below are Chinese retrieval qa prompts

_CUSTOM_SUMMARIZER_TEMPLATE_ZH = """请递进式地总结所提供的当前对话，将当前对话的摘要内容添加到先前已有的摘要上，返回一个融合了当前对话的新的摘要。

例1:
已有的摘要:
人类问Assistant对人工智能的看法。人工智能认为人工智能是一种善的力量。

新的对话内容:
人类: 为什么你认为人工智能是一种好的力量?
Assistant: 因为人工智能将帮助人类充分发挥潜力。

新的摘要:
人类问Assistant对人工智能的看法。人工智能认为人工智能是一种积极的力量，因为它将帮助人类充分发挥潜力。
示例结束

已有的摘要:
{summary}

新的对话内容:
{new_lines}

新的摘要:"""


_ZH_RETRIEVAL_QA_PROMPT = """<指令>根据下列支持文档和对话历史，简洁和专业地来回答问题。如果无法从支持文档中得到答案，请说 “根据已知信息无法回答该问题”。回答中请不要涉及支持文档中没有提及的信息，答案请使用中文。 </指令>

{context}

<对话历史>
{chat_history}
</对话历史>

<问题>{question}</问题>
答案："""

ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS = ["无法回答该问题"]
ZH_RETRIEVAL_QA_REJECTION_ANSWER = "抱歉，根据提供的信息无法回答该问题。"


_ZH_RETRIEVAL_CLASSIFICATION_USE_CASE = """使用提供的参考案例判断客户遇到的故障所属的故障原因分类。

背景信息:
{context}

客服记录:
{question}
故障原因分类："""

_ZH_DISAMBIGUATION_PROMPT = """你是一个乐于助人、恭敬而诚实的助手。你总是按照指示去做。
请用聊天记录中提到的具体名称或实体名称替换给定句子中的任何模糊或有歧义的指代，如果没有提供聊天记录或句子中不包含模糊或有歧义的指代，则只输出原始句子。您的输出应该是消除歧义的句子本身(与“消除歧义的句子:”在同一行中)，并且不包含任何其他内容。

下面是一个例子:
聊天记录:
用户: 我有一个朋友，张三。你认识他吗?
Assistant: 我认识一个叫张三的人

句子: 他最喜欢的食物是什么?
消除歧义的句子: 张三最喜欢的食物是什么?

聊天记录:
{chat_history}

句子: {input}
消除歧义的句子:"""


# Below are English retrieval qa prompts

_EN_RETRIEVAL_QA_PROMPT = """[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist content.
If the answer cannot be inferred based on the given context, please say "I cannot answer the question based on the information given.".<</SYS>>
Use the context and chat history to answer the question.

context:
{context}

chat history
{chat_history}

question: {question}
answer:"""
EN_RETRIEVAL_QA_TRIGGER_KEYWORDS = ["cannot answer the question"]
EN_RETRIEVAL_QA_REJECTION_ANSWER = "Sorry, this question cannot be answered based on the information provided."

_EN_DISAMBIGUATION_PROMPT = """[INST] <<SYS>>You are a helpful, respectful and honest assistant. You always follow the instruction.<</SYS>>
Please replace any ambiguous references in the given sentence with the specific names or entities mentioned in the chat history or just output the original sentence if no chat history is provided or if the sentence doesn't contain ambiguous references. Your output should be the disambiguated sentence itself (in the same line as "disambiguated sentence:") and contain nothing else.

Here is an example:
Chat history:
Human: I have a friend, Mike. Do you know him?
Assistant: Yes, I know a person named Mike

sentence: What's his favorite food?
disambiguated sentence: What's Mike's favorite food?
[/INST]
Chat history:
{chat_history}

sentence: {input}
disambiguated sentence:"""


# Prompt templates

# English retrieval prompt, the model generates answer based on this prompt
PROMPT_RETRIEVAL_QA_EN = PromptTemplate(
    template=_EN_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)
# English disambigate prompt, which replace any ambiguous references in the user's input with the specific names or entities mentioned in the chat history
PROMPT_DISAMBIGUATE_EN = PromptTemplate(template=_EN_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])

# Chinese summary prompt, which summarize the chat history
SUMMARY_PROMPT_ZH = PromptTemplate(input_variables=["summary", "new_lines"], template=_CUSTOM_SUMMARIZER_TEMPLATE_ZH)
# Chinese disambigate prompt, which replace any ambiguous references in the user's input with the specific names or entities mentioned in the chat history
PROMPT_DISAMBIGUATE_ZH = PromptTemplate(template=_ZH_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])
# Chinese retrieval prompt, the model generates answer based on this prompt
PROMPT_RETRIEVAL_QA_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)
# Chinese retrieval prompt for a use case to analyze fault causes
PROMPT_RETRIEVAL_CLASSIFICATION_USE_CASE_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_CLASSIFICATION_USE_CASE, input_variables=["question", "context"]
)
