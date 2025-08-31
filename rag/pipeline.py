import os 
from dotenv import load_dotenv


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.vectorstores import  FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()

#1. llm provider 
class llmsetup:
    def __init__(self):
        api_key=os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("api key not found in env")
        
        self.llm= ChatOpenAI(
            api_key=api_key,
            model="",
            temperature=0.7
        )

    def __call__(self, prompt):

        response= self.llm.invoke(prompt)
        return response 

class Embedder:

    def __init__(self):
        self.Emodel= OpenAIEmbeddings(
                    model='text-embedding-3-small',
                    dimensions=200
        )
    
class Preprocessor:

    def __init__(self, embedder:Embedder):

        self.chunker= SemanticChunker(
            embeddings=embedder.Emodel,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.0
        )    

    def process_pdf(self,pdf_path: str):

        #1 loading pdf into doc obj
        loader= PyPDFLoader(pdf_path)
        docs= loader.load()

        texts= [doc.page_content for doc in docs  ]

        #2. chunkings the text of pdf
        chunk= self.chunker.create_documents(texts)

        return chunk
    

###
###
###callable having main function as run 
    
class RAGPipeline:

    def __init__  (self):
        print("initializing the pipeline")

        self.llm0=llmsetup()
        self.embedder0= Embedder()
        #self.vectordb=VectorDB(self.embedder0)
        self.preprocessor= Preprocessor(self.embedder0)
        #self.retriever= self.vectordb.retriever()

        print("pipeline initialized succsefully")

    
    def run(self, query: str, pdf_path: str =None)->str:

        #1. process the pdf 
        if pdf_path:
            chunks= self.preprocessor.process_pdf(pdf_path)
        
            

        else: 
            raise ValueError("pdf_path is currupt")
        
        #2 retrieve documnets form db

        store= FAISS.from_documents(chunks, self.embedder0 )
        retriever= store.as_retriever(
                            search_type="mmr",
                            search_kwargs= {"k":3, 
                                              "lambda_mult":1}
        )

        retrieved_docs= retriever.invoke(query)
        context= "\n\n".join([i.page_content for i in retrieved_docs])

        #3 prompting and generating final prompt

        prompt= PromptTemplate(
                    template= """ You are a helpful Legal assistant , act as an authentic indian legal scholar.
                                  Answer ONLY from the provided data content , you are allowed to answer authentic and genuine facts also .
                                  If the context is insuficient , just say you dnt know it and context is not availabel in doc.

                                context: {context}  question is {question}

                               """,

                    input_variables= ['context', 'question']
        )

        final_prompt= prompt.invoke({
            "context":context,
            "question":query
        })


        return self.llm0.invoke(final_prompt)
    




    








         

            
        