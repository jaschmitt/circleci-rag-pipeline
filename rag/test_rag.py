
from langchain.prompts              import ChatPromptTemplate
from langchain.chat_models          import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

from dotenv import load_dotenv
import pytest  # -- might need it for fixture, but I assume not



# =================== SETUP =================== #

@pytest.fixture
def thing():
    thing = True
    return thing


# ============================================= #


# =================== TESTS =================== #

def test_hello_world(thing):
    print ("-- other text here")
    assert thing == True


def test_get_response_from_llm():
    load_dotenv()

    template = """You are a helpful assistant who's name is Bob."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0) | StrOutputParser()
    chain.invoke({"text": "What is your name?"})


#def test_get_response_from_llm_2():
#    # load environment which contains API keys
#    load_dotenv()
#
#
#    # define test name
#    test_name = "Bob"
#
#
#    # Create message to feed LLM
#    template = ChatPromptTemplate.from_messages([
#        ("system", "You are a helpful AI bot. Your name is {name}."),
#        ("human",  "Hello, how are you doing?"),
#        ("ai",     "I'm doing well, thanks!"),
#        ("human",  "{user_input}"),
#    ])
#    prompt = template.format_messages(
#        name=test_name,
#    )
#
#
#    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
#    response_generator = (
#        prompt
#        | model
#        | StrOutputParser()
#    )
#
#    chain.invoke({"user_input": "What is your name?"})
#
#
