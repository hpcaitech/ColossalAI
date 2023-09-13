'''
Script for the multilingual conversation based experimental AI agent
We used ChatGPT as the language model
You need openai api key to run this script
'''

from langchain.llms import OpenAI
from langchain.memory import ChatMessageHistory
from langchain.llms import OpenAI
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings.openai import OpenAIEmbeddings
from colossalqa.data_loader.table_dataloader import TableLoader
from colossalqa.data_loader.document_loader import DocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import ZeroShotAgent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.memory.chat_memory import ChatMessageHistory
from langchain import OpenAI, LLMChain
from langchain.vectorstores import Chroma
from langchain.agents.agent import AgentExecutor
import os
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any

# setup openai key
# Set env var OPENAI_API_KEY or load from a file
openai_key = open("/home/lcyab/openai_key.txt").read()
os.environ["OPENAI_API_KEY"] = openai_key

# Load data served on sql
print("Select files for constructing sql database")
tools = []

class CustomLLM(LLM):

    import langchain
    n: int
    llm: langchain.llms.openai.OpenAI

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:

        reply = self.llm(prompt)

        f = open('calls.log', 'a')

        f.write('<request>\n')
        f.write(prompt)
        f.write('</request>\n')

        f.write('<response>\n')
        f.write(reply)
        f.write('</response>\n')
        f.close()

        return reply

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}


# llm = OpenAI(temperature=0.7, verbose=True)
llm = OpenAI(temperature = 0.0)

while True:
    file = input("Select a file to load or enter Esc to exit:")
    if file=='Esc':
        break
    data_name = input("Enter a short description of the data:")

    table_loader = TableLoader([[file, data_name.replace(' ', '_')]], sql_path=f"sqlite:///{data_name.replace(' ', '_')}.db")
    sql_path = table_loader.get_sql_path()

    # create sql database
    db = SQLDatabase.from_uri(sql_path)
    print(db.get_table_info())

    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    name = f"Query the SQL database regarding {data_name}"
    description = f"useful for when you need to answer questions based on data stored on a SQL database regarding {data_name}"
    tools.append(Tool(
        name=name,
        func=db_chain.run,
        description=description,
    ))
    print(f"added sql dataset\n\tname={name}\n\tdescription:{description}")



# VectorDB
embedding = OpenAIEmbeddings()

# Load data serve on sql
print("Select files for constructing retriever")
while True:
    file = input("Select a file to load or enter Esc to exit:")
    if file=='Esc':
        break
    data_name = input("Enter a short description of the data:")
    retriever_data = DocumentLoader([[file, data_name.replace(' ', '_')]]).all_data

    # Split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    splits = text_splitter.split_documents(retriever_data)

    # create vector store
    vectordb = Chroma.from_documents(documents=splits, embedding=embedding)
    # create retriever
    retriever = vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": .5, "k":5})
    # add to tool chain
    name = f"Searches and returns documents regarding {data_name}."
    tools.append(create_retriever_tool(
        retriever, 
        data_name,
        name
    ))
    

prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools. If none of the tools can be used to answer the question. Do not share uncertain answer unless you think answering the question doesn't need any background information. In that case, try to answer the question directly."""
suffix = """You are provided with the following background knowledge:
Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=ChatMessageHistory()
)

llm_chain = LLMChain(llm=OpenAI(temperature = 0.7), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)
agent_chain.set_llm(OpenAI(temperature = 0.2))

while True:
    user_input = input("User: ")
    if ' end ' in user_input:
        print("Agent: Happy to chat with you ：)")
        break
    agent_response = agent_chain.run(user_input)
    print(f"Agent: {agent_response}")
table_loader.sql_engine.dispose() 