"""
All custom prompt templates are defined here.
"""

from langchain.prompts.prompt import PromptTemplate

_CUSTOM_SUMMARIZER_TEMPLATE_ZH = """请递进式地总结所提供的当前对话，将当前对话的摘要内容添加到先前已有的摘要上，返回一个融合了当前对话的新的摘要。

例1:
已有的摘要:
人类问AI对人工智能的看法。人工智能认为人工智能是一种善的力量。

新的对话内容:
人类: 为什么你认为人工智能是一种好的力量?
人工智能: 因为人工智能将帮助人类充分发挥潜力。

新的摘要:
人类问AI对人工智能的看法。人工智能认为人工智能是一种积极的力量，因为它将帮助人类充分发挥潜力。
示例结束

已有的摘要:
{summary}

新的对话内容:
{new_lines}

新的摘要:"""

# _ZH_RETRIEVAL_QA_PROMPT = """你是一个善于解答用户问题的AI助手。在保证安全的前提下，回答问题要尽可能有帮助。你的答案不应该包含任何有害的、不道德的、种族主义的、性别歧视的、危险的或非法的内容。请确保你的回答是公正和积极的。
# 如果不能根据给定的上下文推断出答案，请不要分享虚假、不确定的信息。
# 使用提供的背景信息和聊天记录对用户的输入作出回应或继续对话。您应该只生成一个回复。不需要跟进回答。请使用中文作答。

# 背景信息:
# {context}

# 聊天记录:
# {chat_history}

# 用户: {question}
# AI:"""


_ZH_RETRIEVAL_QA_PROMPT = """<指令>根据下列支持文档和对话历史，简洁和专业地来回答问题。如果无法从支持文档中得到答案，请说 “根据已知信息无法回答该问题”。回答中请不要涉及支持文档中没有提及的信息，答案请使用中文。 </指令>

{context}

<对话历史>
{chat_history}
</对话历史>

<问题>{question}</问题>
<答案>"""

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
AI: 我认识一个叫张三的人

句子: 他最喜欢的食物是什么?
消除歧义的句子: 张三最喜欢的食物是什么?

聊天记录:
{chat_history}

句子: {input}
消除歧义的句子:"""

_EN_RETRIEVAL_QA_PROMPT = """[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If the answer cannot be infered based on the given context, please don't share false information.<</SYS>>
Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

context:
{context}

chat history
{chat_history}

Human: {question}
AI:"""

_EN_DISAMBIGUATION_PROMPT = """[INST] <<SYS>>You are a helpful, respectful and honest assistant. You always follow the instruction.<</SYS>>
Please replace any ambiguous references in the given sentence with the specific names or entities mentioned in the chat history or just output the original sentence if no chat history is provided or if the sentence doesn't contain ambiguous references. Your output should be the disambiguated sentence itself (in the same line as "disambiguated sentence:") and contain nothing else.

Here is an example:
Chat history:
Human: I have a friend, Mike. Do you know him?
AI: Yes, I know a person named Mike

sentence: What's his favorate food?
disambiguated sentence: What's Mike's favorate food?
[/INST]
Chat history:
{chat_history}

sentence: {input}
disambiguated sentence:"""


PROMPT_RETRIEVAL_QA_EN = PromptTemplate(
    template=_EN_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)

PROMPT_DISAMBIGUATE_EN = PromptTemplate(template=_EN_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])

SUMMARY_PROMPT_ZH = PromptTemplate(input_variables=["summary", "new_lines"], template=_CUSTOM_SUMMARIZER_TEMPLATE_ZH)

PROMPT_DISAMBIGUATE_ZH = PromptTemplate(template=_ZH_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])

PROMPT_RETRIEVAL_QA_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)

PROMPT_RETRIEVAL_CLASSIFICATION_USE_CASE_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_CLASSIFICATION_USE_CASE, input_variables=["question", "context"]
)
