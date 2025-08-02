"""# Import Packages"""

import os
import chromadb
import requests
import json
import gradio as gr
import re
import numpy as np

from datasets import load_dataset
from chromadb.config import Settings
from huggingface_hub import hf_hub_download
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, BitsAndBytesConfig

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, Annotated, Union
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END

"""# Read Data"""

import pandas as pd

df = pd.read_parquet("hf://datasets/openai/openai_humaneval/openai_humaneval/test-00000-of-00001.parquet")
df

"""# Embedding Model"""

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},  # Choose based on your setup
    encode_kwargs={"normalize_embeddings": True}  # Important for BGE
)

"""# Chroma VectorDB"""

#HumanEval Dataset
documents1 = [Document(page_content=prom, metadata={"task_id": task_id})
             for prom, task_id in zip(df["prompt"], df['task_id'])]
vectorstore = Chroma.from_documents(documents=documents1, embedding=embedding_model)

#MBPP Dataset
dataset_mbpp = load_dataset("mbpp", split="test[:]")


documents2 = [Document(page_content=text, metadata={"task_id": task_id})
             for task_id, text in zip(dataset_mbpp["task_id"], dataset_mbpp["text"])]
vectorstore.add_documents(documents2)

"""# Retriever"""

retrieved_docs = vectorstore.similarity_search("reverse a string", k=1)

for doc in retrieved_docs:
    print("Prompt:", doc.page_content)
    print("Task ID:", doc.metadata["task_id"])
    print(type(doc.metadata["task_id"]))

"""# LLM Generative Model

"""

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True, quantization_config=BitsAndBytesConfig(load_in_4bit=True), device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#function for generating code Using RAG
def generate_code(query, models=model, tokenizers= tokenizer):

    docs = vectorstore.similarity_search(query, k=1)
    context = [doc.page_content for doc in docs][0]
    task_id = [doc.metadata['task_id'] for doc in docs][0]


    if isinstance(task_id, int):
        if context in dataset_mbpp['text']:
            idx = dataset_mbpp['text'].index(context)
            code = dataset_mbpp['code'][idx]
            test_case = dataset_mbpp['test_list'][idx][0]
            match = re.search(r'assert\s+(\w+)\s*\(', test_case)

            if match:
                function_name = match.group(1)
                print(function_name)

        else:
            code = None
            print("None Issue")
    elif isinstance(task_id, str):

        if context in df['prompt'].values:
            code = df[df['prompt'] == context]["canonical_solution"].values[0]
        else:
            code= None
            print("None Issue")
    else:
        code = None


    prompt = f"""You are a Python code generator. Follow these precise instructions:

    1. Use the following functions in sequence when generating code:
    {code}

    2. This is your function name:
    {function_name}
    3. Test output very well.

    4. Task:
    {query}

    Only use the information provided above to generate Python code. Do not include explanations or any output other than the code itself.

    Your response should start with the Python code directly and end before test cases."""


    messages = [
        {"role": "system", "content": "You are only a Python code generator"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", max_length=500).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response

#function for generating code Using RAG
def generate_code_without_RAG(query, models=model, tokenizers= tokenizer):
    docs = vectorstore.similarity_search(query, k=1)
    context = [doc.page_content for doc in docs][0]
    task_id = [doc.metadata['task_id'] for doc in docs][0]

    if isinstance(task_id, int):
        if context in dataset_mbpp['text']:
            idx = dataset_mbpp['text'].index(context)
            test_case = dataset_mbpp['test_list'][idx][0]
            match = re.search(r'assert\s+(\w+)\s*\(', test_case)

            if match:
                function_name = match.group(1)
                print(function_name)

    prompt = f"""You are a Python code generator. Follow these precise instructions:

    1. This is your function name:
    {function_name}
    2. Test output very well

    3. Task:
    {query}

    Only use the information provided above to generate Python code. Do not include explanations or any output other than the code itself.

    Your response should start with the Python code directly."""


    messages = [
        {"role": "system", "content": "You are only a Python code generator"},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt", max_length=500).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=500, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response

#function for explain code
def explain_code(code, models=model, tokenizers= tokenizer):

    prompt = f"""You are a Python code Explainer.
    Explain the code only without anything else:
------------------
{code}
------------------

Answer:"""

    messages = [
        {"role": "system", "content": "You are a Python code Explainer."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    return response

"""# Evaluate LLM using Correcteness"""

# Remove opening ```python and closing ```
def clean_markdown_code_block(code_block: str) -> str:
    lines = code_block.strip().splitlines()

    # Remove ```python from the top if present
    if lines and lines[0].strip().startswith("```python"):
        lines = lines[1:]

    # Remove trailing ``` if present
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]

    # Join lines and cut off anything after triple single or double quotes
    cleaned_code = "\n".join(lines)
    for triple_quote in ('```'):
        if triple_quote in cleaned_code:
            cleaned_code = cleaned_code.split(triple_quote)[0]

    return cleaned_code.strip()

# add test cases to ensure perfect execution
def evaluate_code_correctness(generated_code, test_list):
    test_code = "\n".join(test_list)
    clean_code = clean_markdown_code_block(generated_code) + "\n" + test_code
    print(clean_code)
    try:
        exec(clean_code)
        print("Excuted")
        return 1  # Correct
    except Exception as e:
        print("ERORRRRRRR")

        return 0  # Incorrect

#Evaluate our LLM withou RAG
queries = dataset_mbpp['text'][0:10]
tests = dataset_mbpp['test_list'][0:10]
acc_without = 0
count_without = 0

for query, test in zip(queries, tests):
    code_generatded_cleaned = clean_markdown_code_block(generate_code_without_RAG(query))
    acc_without += evaluate_code_correctness(code_generatded_cleaned, test[:])
    count_without += 1
    print(f'Executed - {acc_without}')
    print(f'Step number = {count_without}')

#Evaluate our LLM withou RAG
queries = dataset_mbpp['text'][0:10]
tests = dataset_mbpp['test_list'][0:10]
acc = 0
count = 0

for query, test in zip(queries, tests):
    code_generatded_cleaned = clean_markdown_code_block(generate_code(query))
    acc += evaluate_code_correctness(code_generatded_cleaned, test[:])
    count += 1
    print(f'Executed - {acc}')
    print(f'Step number = {count}')

print(f'Correctness of our LLM (without RAG) = {(acc_without / 10) * 100 }%')
print(f'Correctness of our LLM (wit RAG) = {(acc / 10) * 100 }%')

"""# LangGraph"""

class AgentState(TypedDict):
    messages: list[Union[HumanMessage, AIMessage]]

    will_gen: int
    will_exp: int

    query: Annotated[list[HumanMessage], add_messages]
    code: Annotated[list[HumanMessage], add_messages]

    gen: Annotated[list[AIMessage], add_messages]
    exp: Annotated[list[AIMessage], add_messages]

#Using a LLM model, we decide if the input is generation or explaination task
def inputs(state: AgentState) -> AgentState:
    last_msg = state["messages"][-1].content

    # Define chat-style messages
    messages = [
        {"role": "system", "content": (
            "You are a smart classifier that decides if the user's intent is to generate code or explain code. "
            "Respond ONLY with a JSON object in this format:\n"
            '{ "task": "generate" or "explain", "user_input": "<copy of user message>" }'
        )},
        {"role": "user", "content": last_msg}
    ]

    # Call model
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)

    # Extract JSON
    try:
        json_start = response.find("{")
        json_str = response[json_start:]
        task_info = json.loads(json_str)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from model: {raw_output}") from e

    task = task_info["task"]
    user_input = task_info["user_input"]

    if task == "generate":
        state["will_gen"] = 1
        state["will_exp"] = 0
        state["query"] = [HumanMessage(content=user_input)]
    else:
        state["will_gen"] = 0
        state["will_exp"] = 1
        state["code"] = [HumanMessage(content=user_input)]

    return state

def generate(state:AgentState) -> AgentState:
    """This function generate code using query"""
    state['gen'] = [AIMessage(content=generate_code(state["query"][-1].content))]
    state['messages'].append(AIMessage(content=generate_code(state["query"][-1].content)))
    return state

def explain(state:AgentState) -> AgentState:
    '''This function explain the code'''
    state['exp'] = [AIMessage(content=explain_code(state["code"][-1].content))]
    return state

def decide_next_node(state:AgentState) -> AgentState:
    '''This function decide to generate or explain the input'''
    if state["will_gen"] == 1:
        return "Generating"
    else:
        return "Explaining"

graph = StateGraph(AgentState)

graph.add_node("Input", inputs)
graph.add_node("generate_code", generate)
graph.add_node("explain_code", explain)
graph.add_node("router1", lambda state: state)


graph.add_edge(START, "Input")
graph.add_edge("Input", "router1")

graph.add_conditional_edges(
    "router1",
    decide_next_node,
    {
        "Generating": "generate_code",
        "Explaining": "explain_code"
    }

)

graph.add_edge("generate_code", END)
graph.add_edge("explain_code", END)

app = graph.compile()

"""# Schema"""

from IPython.display import Image, display
display(Image(app.get_graph().draw_mermaid_png()))

"""# Gradio Deployment"""

chat_history = []
explain_generated = False
def chat_with_bot(user_input, history):
    global chat_history, explain_generated

    if explain_generated == True:
        if user_input.lower() == 'yes':
            result = app.invoke({"messages" : chat_history, "will_exp": 1})
            explain_generated = False
            return result['exp'][-1].content
        else:
            explain_generated = False
            return 'Tell me what you need'



    chat_history.append(HumanMessage(content=user_input))
    result = app.invoke({"messages" : chat_history})
    chat_history = result['messages']

    if result['will_gen'] == 1:
        explain_generated = True
        return result['gen'][-1].content + "\n\n" + "Do you want from me to explain the code?"

    elif result['will_exp'] == 1:
        explain_generated = False
        return result['exp'][-1].content


gr.ChatInterface(
    fn=chat_with_bot,
    title="Code Helper Bot",
    description="Ask me to write or explain code. I can also explain generated code if you want!",
    theme="soft"
).launch()