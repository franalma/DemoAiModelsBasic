import os
from dotenv import load_dotenv, find_dotenv
from open_ai_helper import OpenAI_Helper
from pine_cone_helper import PineCone_Helper
from lang_chain_helper import LangChainHelper
import numpy as np


ai_helper = OpenAI_Helper()
pine_cone_helper = PineCone_Helper()
lang_helper = LangChainHelper()
PC_INDEX_NAME = 'langchain-index'


def load_open_ai_key():
    load_dotenv(find_dotenv(), override=True)
    return os.environ.get("OPENAI_API_KEY")

def load_pineconde_key():
    load_dotenv(find_dotenv(), override=True)
    return os.environ.get("PINECONE_API_KEY")


def init ():
    pine_cone_helper.init(load_pineconde_key())
    ai_helper.init(load_open_ai_key())
    
    

def split_file(): 
    with open ('solar_text.txt') as f: 
        fileContent = f.read()
    values = lang_helper.split(200, 20, fileContent)
    return values
    
    
# def do_embed(values):
#     lang_helper.print_estimate_embbeding_cost(model="text-embedding-ada-002", texts=values)
#     result =[len(values)]
#     for i in range (0,len(values)):        
#         temp = lang_helper.create_embbedings(values[i].page_content)
#         result[i] = (temp)
#     return result
    
    
def create_embeddings():

    # pine_cone_helper.create_index(PC_INDEX_NAME)
    # embedded = do_embed(input_value)
    input_value = split_file()
    lang_helper.put_embbeded(input_value, PC_INDEX_NAME)

def run_question_without_llm(question,option):
    # emb = lang_helper.create_embbedings(text=question)
    # response = pine_cone_helper.search(query_vector=emb, index_name=PC_INDEX_NAME)
    if option ==0:
        lang_helper.find_similarity_with_openai(input=question, index_name=PC_INDEX_NAME)
    else:
        lang_helper.find_similarity_with_hugging_face(input=question, index_name=PC_INDEX_NAME)
  
def run_question_with_llm(question, option):
    if (option == 0):    
        response = lang_helper.generate_interaction_with_openai(input=question, index_name=PC_INDEX_NAME) 
    else:
        response = lang_helper.generate_interaction_with_hugging(input=question, index_name=PC_INDEX_NAME) 
    print (response)

def load_file(path, option):
    print(path)
    name, extension = os.path.splitext(path)
    if option == 0:
        lang_helper.load_file(name=name, extension=extension, index_name=PC_INDEX_NAME)
    elif option == 1:
        lang_helper.load_file_embedding_hugging_face(name=name, extension=extension, index_name=PC_INDEX_NAME)
    
    
    
def print_menu():
    while(True):
        print("========================")
        print("1. Init (optional)")
        print("2. Create embeddings for solar.txt")
        print("3. Use exisiting index without LLM (openAI) ")
        print("3.1. Use exisiting index without LLM (HF)")
        print("4. Use exisiting index with OpenAI LLM")
        print ("5. Load pdf file to db")
        print("6. Create PineCone index (openAI=1536; hugging = 768)")
        print("7. Delete PineCone index")
        print("8. Create embedding with Hugging Face model")
        print("9. Use exisiting index with Hugging Face LLM")
        print ("10. Load pdf from directory")
        print ("0. Exit")
        value= input("Seleccine una opción:")
        
        if value == "1":
            init()
        elif value == "2":
            create_embeddings()
        elif value == "3":
            question = input ("Introduce una pregunta: ")
            run_question_without_llm(question,0)
        elif value == "3.1":
            question = input ("Introduce una pregunta: ")
            run_question_without_llm(question,1)
        elif value == "4":
            question = input ("Introduce una pregunta: ")
            run_question_with_llm(question,0)    
        elif value == "5":
            path = input ("Introduce el path del archivo pdf:  ")            
            load_file(path, 0)
        elif value == "6":
            index_size  = input("Introduce la longitud del índice: ")
            pine_cone_helper.create_index(PC_INDEX_NAME, int(index_size))
        elif (value == "7"):
            pine_cone_helper.remove_index()
        elif (value == "8"):
            path = input("Introduce el nombre de fichero: ")      
            load_file(path, 1)
        elif (value == "9"):
            question = input ("Introduce una pregunta: ")    
            run_question_with_llm(question,1)    
        elif (value == "10"):
            path = input ("Introduce el path del directorio: ")
            list = os.listdir(path)
            files = []
            for f in list:
                if os.path.isfile(f):
                    files.insert(f)
                else: 
                    list.append(os.listdir(f))                    
                
                
            
            print (files)            
        elif value == 0:
            break
        
                    
init()
print_menu() 







