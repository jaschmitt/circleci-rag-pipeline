
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

@pytest.fixture
def chain():
    # Load environment (API keys)
    load_dotenv()

    # Define chain
    template = """You are a helpful assistant who's name is Bob."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain = chat_prompt | ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0) | StrOutputParser()

    return chain


# ============================================= #




# =================== TESTS =================== #

def test_hello_world(thing):
    print ("-- other text here")
    assert thing == True


def test_get_response_from_llm(chain):
    # Define input/output
    input_text  = "What is your name?"
    output_text = chain.invoke({"text": input_text})
    print("Question: " + input_text)
    print("Answer:   " + output_text)
    
    assert "bob" in output_text.lower()


