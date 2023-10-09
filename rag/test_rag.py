# LangChain libraries
from langchain.document_loaders       import RecursiveUrlLoader
from langchain.document_transformers  import Html2TextTransformer
from langchain.text_splitter          import TokenTextSplitter
from langchain.embeddings             import OpenAIEmbeddings
from langchain.vectorstores           import Chroma
from langchain.prompts                import ChatPromptTemplate
from langchain.chat_models            import ChatOpenAI
from langchain.schema.output_parser   import StrOutputParser
from langchain.smith                  import RunEvalConfig, run_on_dataset

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
def chain_2():
    # Load environment (API keys)
    load_dotenv()

    # Load in langsmith documentation as test
    api_loader      = RecursiveUrlLoader("https://docs.smith.langchain.com")
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
            "context": itemgetter("question") | retriever | (lambda docs: "\n".join([doc.page_content for doc in docs])),
            "question": itemgetter("question")
        }
        | response_generator
    )
    
    return chain_2



# ============================================= #




# =================== TESTS =================== #

def test_name(chain_1):
    # instantiate LangSmith client
    client = Client()

    # Define input/output
    input_text  = "What is your name?"
    output_text = chain_1.invoke({"text": input_text})
    print("Question: " + input_text)
    print("Answer:   " + output_text)
    
    assert "bob" in output_text.lower()


def test_basic_arithmetic(chain_1):
    # instantiate LangSmith client
    client = Client()

    # Define input/output
    input_text  = "What is your 5 + 7?"
    output_text = chain_1.invoke({"text": input_text})
    print("Question: " + input_text)
    print("Answer:   " + output_text)
    
    assert "12" in output_text.lower()


def test_documentation_llm_evaluators(chain_2):
    # instantiate LangSmith client
    client = Client()


    # define question and expected output example test set for qa evaluators in langsmith.
    examples = [
        ("what is langchain?", "langchain is an open-source framework for building applications using large language models. it is also the name of the company building langsmith."),
        ("how might i query for all runs in a project?", "client.list_runs(project_name='my-project-name'), or in typescript, client.listruns({projectname: 'my-project-anme'})"),
        ("what's a langsmith dataset?", "a langsmith dataset is a collection of examples. each example contains inputs and optional expected outputs or references for that data point."),
        ("how do i use a traceable decorator?", """the traceable decorator is available in the langsmith python sdk. to use, configure your environment with your api key,\
    import the required function, decorate your function, and then call the function. below is an example:
    ```python
    from langsmith.run_helpers import traceable
    @traceable(run_type="chain") # or "llm", etc.
    def my_function(input_param):
        # function logic goes here
        return output
    result = my_function(input_param)
    ```"""),
        ("can i trace my llama v2 llm?", "so long as you are using one of langchain's llm implementations, all your calls can be traced"),
        ("why do i have to set environment variables?", "environment variables can tell your langchain application to perform tracing and contain the information necessary to authenticate to langsmith."
         " while there are other ways to connect, environment variables tend to be the simplest way to configure your application."),
        ("how do i move my project between organizations?", "langsmith doesn't directly support moving projects between organizations.")
    ]

    # create a dataset for langsmith evaluator using the example qa list from above
    dataset_name = f"retrieval qa questions {str(uuid.uuid4())}"
    dataset = client.create_dataset(dataset_name=dataset_name)
    for q, a in examples:
        client.create_example(inputs={"question": q}, outputs={"answer": a}, dataset_id=dataset.id)


    # run qa evaluators
    eval_config = RunEvalConfig(
        evaluators=["qa"],
        # if you want to configure the eval llm:
        # eval_llm=chatanthropic(model="claude-2", temperature=0)
    )


#    # determine project name
#    project_name = os.environ.get("LANGCHAIN_PROJECT") + "_evaluators"


    # Run the evaluation. This makes predictions over the dataset and then uses
    # the "QA" evaluator to check the correctness on each data point.
    _ = client.run_on_dataset(
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda: chain_2,
        evaluation=eval_config,
#        project_name=project_name,
    )


