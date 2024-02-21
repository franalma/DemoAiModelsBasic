from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModel,AutoModelForCausalLM, LlamaTokenizer,AutoModelForQuestionAnswering
from transformers import AutoTokenizer, pipeline

class HuggingFaceHelper:
    
    def load_hugging_face_embbeding(self):
        # model_name = "sentence-transformers/all-MiniLM-L6-v2"
        # model_kwargs = {'device': 'cpu'}
        # encode_kwargs = {'normalize_embeddings': False}
        # hf = HuggingFaceEmbeddings(
        #     model_name=model_name,
        #     model_kwargs=model_kwargs,
        #     encode_kwargs=encode_kwargs
        # )
        # return hf
        model_name = "BAAI/bge-small-en"
        # model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            # model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf
        
    
    def load_hugging_face_llm(self):
        # Specify the model name you want to use
        model_name = "Intel/dynamic_tinybert"
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)
        question_answerer = pipeline(
            "question-answering", 
            model=model_name, 
            tokenizer=tokenizer,
            return_tensors='pt'
        )   
        llm = HuggingFacePipeline(
            pipeline=question_answerer,
            model_kwargs={"temperature": 0.7, "max_length": 512},
        )
                    
        return llm


        
        # tokenizer = AutoTokenizer.from_pretrained("model/google/flan-t5-large")
        # model = AutoModel.from_pretrained("model/google/flan-t5-large")
        # llm = HuggingFacePipeline(
        # pipeline = pipeline,
        # model_kwargs={"temperature": 0, "max_length": 512},  
        
        # Load the tokenizer associated with the specified model
        # tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

        # Define a question-answering pipeline using the model and tokenizer
        
        
        # llm = pipeline("text-generation", 
        #         model=model, 
        #         tokenizer=tokenizer,
        #         device_map="cpu",
        #         max_new_tokens=1024, 
        #         return_full_text=True,
        #         repetition_penalty=1.1
        #        )

        # Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
        # with additional model-specific arguments (temperature and max_length)
     
    
        