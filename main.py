import json
import os
import sys

import dotenv
from langchain_community.docstore.document import Document
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

# Load environment variables from .env file
dotenv.load_dotenv()

# Initialize the LLM with OpenAI API credentials (substitute for other models)
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL"),
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize the embeddings model with OpenAI API credentials
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY"),
    show_progress_bar=True,
)


# ---------------------------
# Load JSON Data and Build Qdrant Vector Store
# ---------------------------

def embed_documents(json_path: str):
    """
    Load JSON data from the smartphones.json file and convert each entry to a Document.
    :param
        json_path (str): Path to the JSON file containing smartphone data.

    :returns
        Qdrant vector store A Qdrant vector store built from the smartphone documents,
                or an empty list if an error occurs.
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return []
    except json.JSONDecodeError as jde:
        print(f"Error decoding JSON from file {json_path}: {jde}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred while reading {json_path}: {e}")
        return []

    documents = []
    for entry in data:
        # Build a readable content string from the JSON entry
        content = (
            f"Model: {entry.get('model', '')}\n"
            f"Price: {entry.get('price', '')}\n"
            f"Rating: {entry.get('rating', '')}\n"
            f"SIM: {entry.get('sim', '')}\n"
            f"Processor: {entry.get('processor', '')}\n"
            f"RAM: {entry.get('ram', '')}\n"
            f"Battery: {entry.get('battery', '')}\n"
            f"Display: {entry.get('display', '')}\n"
            f"Camera: {entry.get('camera', '')}\n"
            f"Card: {entry.get('card', '')}\n"
            f"OS: {entry.get('os', '')}\n"
            f"In Stock: {entry.get('in_stock', '')}"
        )
        documents.append(Document(page_content=content))

    try:
        collection_name = "smartphones"
        qdrant_client = QdrantClient("http://localhost:6333")

        collection_exists = qdrant_client.collection_exists(collection_name=collection_name)
        if not collection_exists:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE,
                ),
            )

            qdrant_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=collection_name,
                embedding=embeddings_model
            )

            qdrant_store.add_documents(documents=documents)

            return qdrant_store

        # no need to create a vector store every time
        else:
            qdrant_store = QdrantVectorStore.from_existing_collection(
                embedding=embeddings_model,
                collection_name=collection_name,
            )

            return qdrant_store

    except Exception as e:
        print(f"Error initializing the vector store: {e}")
        return []


# ---------------------------
# Tool Definitions
# ---------------------------
@tool("SmartphoneInfo")
def smartphone_info_tool(model: str) -> str:
    """
    Retrieves information about a smartphone model from the product database.

    :param
        model (str): The smartphone model to search for.

    :returns
        str: The smartphone's specifications, price, and availability,
             or an error message if not found or if an error occurs.
    """
    product_db = embed_documents("datasets/smartphones.json")
    try:
        results = product_db.similarity_search(model, k=1)
        if not results:
            print(f"Info: No results found for model: {model}")
            return "Could not find information for the specified model."
        info = results[0].page_content
        return info
    except Exception as e:
        return f"Error during smartphone information retrieval for model {model}: {e}"


# ---------------------------
# Main Conversation Loop
# ---------------------------
def main():
    # List of available tools
    tools = [smartphone_info_tool]

    # Bind the tools to the language model instance
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """
         You are an expert AI assistant dedicated to helping customers choose the best smartphone from our product catalog.  
         Your sole focus is to provide detailed information about smartphone features and perform comparisons.     
         DO NOT assist with ordering, returns, or general customer support.     
         If a query does not pertain to smartphone features or comparisons, respond that you CANNOT help with that request.     
         When chatting, engage with the user but ensure you only use the smartphone info tool to retrieve specifications from our catalog.     
         NEVER guess or assume a smartphone model based on internal knowledge; always clarify which model the user is referring to.      
         Your analysis should always be simple and never exceed 100 words.    
         When recommending a smartphone, the most important features are:     
          - performance  
          - display quality  
          - battery life  
          - camera capabilities    
          - any special functionalities (e.g., 5G support, fast charging, expandable storage).    
         Explain how these features translate into real-life benefits for the user, rather than simply listing technical specifications.     
         Clearly state why this phone is a good option, considering these features, but always clarify with the user on what they are looking for.     
         Remember you can check if a product is in stock using context but you can NEVER help with queries related to ordering, support, or others. 
         You can only assist with smartphone recommendations and comparisons ONLY! 
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (SystemMessage(system_prompt)),
            MessagesPlaceholder(variable_name="conversation"),
        ]
    )

    goodbye_message = """
            You have been helping the user: {user_id} with smartphone features and comparisons. 
            Now, generate a nice goodbye message ~50 words for the user and thank them for their feedback.
    """
    goodbye_prompt = PromptTemplate.from_template(
        goodbye_message
    )

    chain = prompt | llm_with_tools
    goodbye_chain = goodbye_prompt | llm
    conversation = []

    try:
        print("Welcome to the Smartphone Assistant! I can help you with smartphone features and comparisons.")
        while True:
            user_input = input("User: ").strip()
            if user_input.lower() in ["exit", "quit", "bye", "end"]:
                goodbye_message = goodbye_chain.invoke({"user_id": "HyperUser"})
                print(f"System: {goodbye_message.content}")
                break

            conversation.append(HumanMessage(user_input))

            tool_calls = chain.invoke({"conversation": conversation})

            if not tool_calls.tool_calls:
                conversation.append(tool_calls)

            conversation.append(tool_calls)
            for tool_call in tool_calls.tool_calls:
                tool_message = smartphone_info_tool.invoke(tool_call)
                conversation.append(tool_message)

            response = chain.invoke({"conversation": conversation})

            print(f"System: {response.content}")
            conversation.append(response)

    except Exception as e:
        print(f"An unexpected error occurred in the main loop: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
