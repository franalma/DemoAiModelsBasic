import gradio as gr
from lang_chain_helper import LangChainHelper
import os
from dotenv import load_dotenv, find_dotenv
from open_ai_helper import OpenAI_Helper
from pine_cone_helper import PineCone_Helper

load_dotenv(find_dotenv(), override=True)
pine_cone_key = os.environ.get("PINECONE_API_KEY")
pine_cone_helper = PineCone_Helper()
pine_cone_helper.init(pine_cone_key)
helper = LangChainHelper()



def process_message(message, history):
    result = helper.generate_interaction_with_hugging(message)
    return result

demo = gr.ChatInterface(fn=process_message,  title="ChatBot")
demo.launch(share=True)   
