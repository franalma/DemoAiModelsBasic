from langchain.text_splitter import RecursiveCharacterTextSplitter as RC
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from hugging_face_helper import HuggingFaceHelper
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_community.llms import Ollama
import tiktoken


class LangChain_Helper:
    docsearch: None
    
    def split(self, chunk_size, overlap, text):
        text_splitter = RC(chunk_size = chunk_size, chunk_overlap = overlap, length_function = len)
        values = text_splitter.create_documents([text])
        print(len(values))
        return values
    
    def split_documents(self, chunk_size, overlap, data):
        text_splitter = RC(chunk_size = chunk_size, chunk_overlap = overlap, length_function = len)
        values = text_splitter.split_documents(data)    
        return values
    

    def print_estimate_embbeding_cost(self, model, texts):
        enc = tiktoken.encoding_for_model(model)
        total_tokens = sum ([len(enc.encode(page.page_content)) for page in texts])
        print (f'Total_tokens:{total_tokens}')
        print (f'Embedding_cost USD:{total_tokens/1000*0.0004:.6f}')

    def create_embbedings(self, text):
        embeddings = OpenAIEmbeddings()
        embedded = embeddings.embed_query(text)
        return embedded
    
    
    def put_embbeded_hugging(self, texts, index):
        embeddings = HuggingFaceHelper().load_hugging_face_embbeding()        
        Pinecone.from_documents(texts, embeddings, index_name =index)
        
    def put_embbeded (self, texts, index):
        embeddings = OpenAIEmbeddings()
        self.docsearch = Pinecone.from_documents(texts, embeddings, index_name =index)
        
    def find_similarity(self,input,index_name):
        embedding = OpenAIEmbeddings()
        db = Pinecone.from_existing_index(index_name,embedding)
        # vectorstore = Pinecone(index_name=index_name, embedding=embedding)
        # result = vectorstore.similarity_search(input)    
        # result = Pinecone.similarity_search_by_vector(embedding=openai_embs)
        result = db.similarity_search(input)
        print (result)
        
        
    def generate_interaction_with_openai(self, input, index_name):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=1)
        embedding = OpenAIEmbeddings()
        db = Pinecone.from_existing_index(index_name,embedding)
        retriever = db.as_retriever (search_type="similarity", search_kwargs={'k':3})
        chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        response = chain.run(input)
        return response
    
    def generate_interaction_with_hugging(self, input, index_name):
        hugging_helper = HuggingFaceHelper()
        # llm = hugging_helper.load_hugging_face_llm()
        llm = Ollama(model="mistral")
        embedding = hugging_helper.load_hugging_face_embbeding()
        response = None
        
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        db = Pinecone.from_existing_index(index_name,embedding)
        try:
            retriever = db.as_retriever (search_kwargs={'k':4})
            chain = (
                {"context": retriever,"question":RunnablePassthrough()}
                |prompt
                |llm
                |StrOutputParser()                                      
                )
            # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="refine", retriever=retriever, return_source_documents=False)
            response = chain.invoke(input)
            return response
        except Exception as error:
            print(error)
            
    
    def load_file_embedding_openai(self, name, extension, index_name):
        result = None
        
        if (extension == ".pdf"):
            loader = PyPDFLoader(name + extension)
            result = loader.load()
            print (f"N pages: {len(result)}")
            fragments = self.split_documents(chunk_size=150, overlap=20, data=result)
            self.print_estimate_embbeding_cost(model="text-embedding-ada-002",texts=fragments)
            # self.put_embbeded(texts=fragments, index=index_name)                    
            self.put_embbeded_hugging(texts=fragments, index=index_name)
        else:
            print("Archivo no soportado")
        
        return result            
    
    def load_file_embedding_hugging_face(self, name, extension, index_name):
        result = None
        
        if (extension == ".pdf"):
            loader = PyPDFLoader(name + extension)
            result = loader.load()
            print (f"N pages: {len(result)}")
            fragments = self.split_documents(chunk_size=150, overlap=20, data=result)
            # self.put_embbeded(texts=fragments, index=index_name)                    
            self.put_embbeded_hugging(texts=fragments, index=index_name)
        else:
            print("Archivo no soportado")
        
        return result    