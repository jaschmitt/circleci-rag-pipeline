# LangChain libraries
from langchain.document_loaders       import RecursiveUrlLoader
from langchain.document_transformers  import Html2TextTransformer
from langchain.text_splitter          import TokenTextSplitter
from langchain.embeddings             import OpenAIEmbeddings
from langchain.vectorstores           import Chroma
from langchain.prompts                import ChatPromptTemplate
from langchain.chat_models            import ChatOpenAI
from langchain.schema.output_parser   import StrOutputParser
from langchain.smith                  import RunEvalConfig

# LangSmith libraries
from langsmith import Client

# Other libraries
from   datetime import datetime
from   dotenv   import load_dotenv
import pytest
import uuid
from   operator import itemgetter
import os



# =================== SETUP =================== #

@pytest.fixture
def thing():
    thing = True
    return thing

@pytest.fixture
def chain_1():
    # Load environment (API keys)
    load_dotenv()

    # Define chain_1
    template = """You are a helpful assistant who's name is Bob."""
    human_template = "{text}"

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", template),
        ("human", human_template),
    ])
    chain_1 = chat_prompt | ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0) | StrOutputParser()

    return chain_1


@pytest.fixture
def chain_2(vector_store):
    # Load environment (API keys)
    load_dotenv()

    # Define chain_2
    prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful documentation Q&A assistant, trained to answer"
                " questions from LangSmith's documentation."
                " LangChain is a framework for building applications using large language models."
                "\nThe current time is {time}.\n\nRelevant documents will be retrieved in the following messages."),
                ("system", "{context}"),
                ("human","{question}")
            ]
        ).partial(time=str(datetime.now()))

    model = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
    response_generator = (
        prompt
        | model
        | StrOutputParser()
    )

    chain_2 = (
        # The runnable map here routes the original inputs to a context and a question dictionary to pass to the response generator
        {
            "context": itemgetter("question") | vector_store | (lambda docs: "\n".join([doc.page_content for doc in docs])),
            "question": itemgetter("question")
        }
        | response_generator
    )
    
    return chain_2


@pytest.fixture
def vector_store():
    # Load environment (API keys)
    load_dotenv()

    document_url   = os.environ.get("LANGCHAIN_ENDPOINT")

    # Loader
    api_loader      = RecursiveUrlLoader(document_url)
    raw_documents   = api_loader.load()

    # Transformer
    doc_transformer = Html2TextTransformer()
    transformed     = doc_transformer.transform_documents(raw_documents)

    # Splitter
    text_splitter  = TokenTextSplitter(
        model_name="gpt-3.5-turbo",
        chunk_size=2000,
        chunk_overlap=200,
    )
    documents  = text_splitter.split_documents(transformed)

    # Define vector store based on documents
    embeddings   = OpenAIEmbeddings()
    vectorstore  = Chroma.from_documents(documents, embeddings)
    retriever    = vectorstore.as_retriever(search_kwargs={"k": 4})

    return retriever

# ============================================= #




# =================== TESTS =================== #

def test_hello_world(thing):
    print ("-- other text here")
    assert thing == True


def test_name(chain_1):
    # Define input/output
    input_text  = "What is your name?"
    output_text = chain_1.invoke({"text": input_text})
    print("Question: " + input_text)
    print("Answer:   " + output_text)
    
    assert "bob" in output_text.lower()


def test_basic_arithmetic(chain_1):
    # Define input/output
    input_text  = "What is your 5 + 7?"
    output_text = chain_1.invoke({"text": input_text})
    print("Question: " + input_text)
    print("Answer:   " + output_text)
    
    assert "12" in output_text.lower()


def test_documentation_correctness(chain_2):
    for tok in chain_2.stream({"question": "How do I log user feedback to a run?"}):
        print(tok, end="", flush=True)


    eval_config = RunEvalConfig(
        evaluators=["qa"],
        # If you want to configure the eval LLM:
        # eval_llm=ChatAnthropic(model="claude-2", temperature=0)
    )

    """Run the evaluation. This makes predictions over the dataset and then uses the "QA" evaluator to check the correctness on each data point."""

#    _ = await client.arun_on_dataset(
#        dataset_name=dataset_name,
#        llm_or_chain_factory=lambda: chain_2,
#        evaluation=eval_config,
#    )


#    # Define input/output
#    input_text  = "What is your name?"
#    output_text = chain_1.invoke({"text": input_text})
#    print("Question: " + input_text)
#    print("Answer:   " + output_text)

    

