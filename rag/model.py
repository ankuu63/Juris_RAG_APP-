from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import  FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

import os 
from dotenv import load_dotenv
load_dotenv()



class VectorDB:
    #def __init__(self,embedder: Embedder):
        #self.store= FAISS(
            #embedding_function= embedder.Emodel,
          # index=
        #)
    
    def retriever(self):
        retriever= self.store.as_retriever(
                            search_type="mmr",
                            search_kwargs= {"k":3, 
                                              "lambda_mult":1}
        )
        return retriever
    
    
    def add_documents_inVDB(self, docs):
        self.store.add_documents(docs)


class RAgpipeline:

                ##
                ##loading during pipeline called in backend 
    def __init__(self):
        print ("Entering in Ragpipeline...")

        self.llm= self.load_llm()
        self.embedd= self.load_embedd_model()
        self.splitter= self.load_splitter()
        self.Vstore= self.load_vstore()
        self.retriever= self.load_retriever()
        

        print("pipeline initialized ")

                ###methods
        
    def load_llm(self):
        api_key=os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_key_notfound")
        

        llm= ChatOpenAI(
            api_key=api_key,
            model="",
            temperature=0.7
        )
        return llm
    
    def load_embedd_model(self):
        Emodel= HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        return Emodel
    
    def load_splitter(self):
        splitter= SemanticChunker(
                    embeddings=self.embedd,
                    breakpoint_threshold_type= "standard_deviation",
                    breakpoint_threshold_amount= 1.0
        )
        return splitter
    
    def load_Vstore(self):
        vector_store= FAISS(
                    embedding_function= self.embedd                 ##0or we will directly add docs in FAiss.from_document(chunks, embedd_modekl)
        )
        return vector_store

    def load_retriever(self):
        retriever= self.Vstore.as_retriever(
            search_type="mmr",
            search_kwargs= { "k":3,      #no of topresult
                             "lambda_mult":1  }
        )
        return retriever
    

    
                ##HELPER FUNCTIONS : BELOW

    def preprocess(self,pdf_path: str=None )-> None :

        #1 load the pdf data
        if pdf_path:
            loader= PyPDFLoader(pdf_path)
            docs= loader.load()     ##list of doc obj
        else :
            raise ValueError("Pdf_path not found")
        
        #2 list of doc obj to list of string
        docs_to_listofstring = [doc.page_content for doc in docs]  

        #3 chunking by schemantic chunker
        chunks = self.splitter.create_documents(docs_to_listofstring)

        #4 addding chunks into Vstore
        self.Vstore.add_documents(chunks)

    def contextretrieving(self, querry: str=None )-> str :
        if querry:
            retrieved_doclist =self.retriever.invoke(querry)
        else :
            raise ValueError("qerry is absent") 
        
        #concatenating the retriened docobj pagecontent
        context_doc= "\n\n".join(doc.page_content for doc in retrieved_doclist) 

        return context_doc


    ##making prmopts and final prompts and response generation
    def run(self, querry:str=None, pdf_path: str=None )-> str :
        if pdf_path:
            self.preprocess(pdf_path)   ##pdf embedding -> vectorstore
        else :
            raise ValueError("error in genrating pdf embedding and storation")
        
        #context_doc retrieving
        context_doc= self.contextretrieving(querry)


        prompt= PromptTemplate(
                    template= """ You are a helpful Legal assistant , act as an authentic indian legal scholar.
                                  Answer ONLY from the provided data content , you are allowed to answer authentic and genuine facts also .
                                  If the context is insuficient , just say you dnt know it and context is not availabel in doc.

                                context: {context}  question is {question}

                               """,

                    input_variables= ['context', 'question']
        )
        
        final_prompt= prompt.invoke({
            "context": context_doc,
            "question":  querry
            })

        
        response= self.llm.invoke(final_prompt)
        return response.content





















