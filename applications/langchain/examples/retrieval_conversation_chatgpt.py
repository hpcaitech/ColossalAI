'''
Multilingual retrieval based conversation system backed by ChatGPT
'''
from colossalqa.local.llm import VllmLLM
from colossalqa.data_loader.document_loader import DocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import LLMChain
from langchain.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from colossalqa.retriever import CustomRetriever
from langchain.chains import RetrievalQA
from colossalqa.memory import ConversationBufferWithSummary
from langchain.llms import OpenAI
import os

# setup openai key
# Set env var OPENAI_API_KEY or load from a file
openai_key = open("/home/lcyab/openai_key.txt").read()
os.environ["OPENAI_API_KEY"] = openai_key

llm = OpenAI(temperature = 0.6)

information_retriever = CustomRetriever()
# VectorDB
embedding =  HuggingFaceEmbeddings(model_name="moka-ai/m3e-base",
                           model_kwargs={'device': 'cpu'},encode_kwargs={'normalize_embeddings': False})

# define memory with summarization ability
memory = ConversationBufferWithSummary(llm=llm)

# Load data to vector store
print("Select files for constructing retriever")
documents = []
while True:
    file = input("Select a file to load or enter Esc to exit:")
    if file=='Esc':
        break
    data_name = input("Enter a short description of the data:")
    retriever_data = DocumentLoader([[file, data_name.replace(' ', '_')]]).all_data

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splits = text_splitter.split_documents(retriever_data)
    documents.extend(splits)
# create vector store
vectordb = Chroma.from_documents(documents=documents, embedding=embedding)
print(vectordb._collection.count())
# create retriever    
retriever=vectordb.as_retriever(search_kwargs={"k":3})
information_retriever.set_retriever(retriever=retriever)
information_retriever.set_k(k=3)



prompt_template = """Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If the answer cannot be infered based on the given context, please don't share false information.
Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

context:
{context}

chat history
{chat_history}

Human: {question}
AI:"""

prompt_template_disambiguate = """You are aEsc helpful, respectful and honest assistant. You always follow the instruction.
Please replace any ambiguous references in the given sentence with the specific names or entities mentioned in the chat history or just output the original sentence if no chat history is provided or if the sentence doesn't contain ambiguous references. Your output should be the disambiguated sentence itself (in the same line as "disambiguated sentence:") and contain nothing else.

Here is an example:
Chat history:
Human: I have a friend, Mike. Do you know him?
AI: Yes, I know a person named Mike

sentence: What's his favorate food?
disambiguated sentence: What's Mike's favorate food?
END OF EXAMPLE

Chat history:
{chat_history}

sentence: {input}
disambiguated sentence:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["question", "chat_history", "context"]
)

memory.initiate_document_retrieval_chain(llm, PROMPT, information_retriever, 
    chain_type_kwargs={'chat_history':'', })


PROMPT_DISAMBIGUATE = PromptTemplate(
    template=prompt_template_disambiguate, input_variables=["chat_history", "input"]
)

llm_chain = RetrievalQA.from_chain_type(llm=llm, verbose=False, chain_type="stuff", retriever=information_retriever, 
                                        chain_type_kwargs={"prompt": PROMPT,"memory":memory })
llm_chain_disambiguate = LLMChain(llm=llm, prompt=PROMPT_DISAMBIGUATE)

def disambiguity(input):
    out = llm_chain_disambiguate.run({'input': input, 'chat_history':memory.buffer})
    return out.split('\n')[0]

information_retriever.set_rephrase_handler(disambiguity)

# build MOE
while True:
    user_input = input("User: ")
    print(f"User: {user_input}")
    if ' end ' in user_input:
        print("Agent: Happy to chat with you ：)")
        break    
    agent_response = llm_chain.run(user_input)
    agent_response = agent_response.split('\n')[0]
    print(f"Agent: {agent_response}")
