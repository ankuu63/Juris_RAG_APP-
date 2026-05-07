import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker

load_dotenv()


# =========================
# LLM SETUP
# =========================

class llmsetup:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("API key not found in env")

        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            temperature=0.7
        )

    def __call__(self, prompt):

        response = self.llm.invoke(prompt)

        return response.content


# =========================
# EMBEDDING MODEL
# =========================

class Embedder:

    def __init__(self):

        self.Emodel = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


# =========================
# PDF PREPROCESSOR
# =========================

class Preprocessor:

    def __init__(self, embedder: Embedder):

        self.chunker = SemanticChunker(
            embeddings=embedder.Emodel,
            breakpoint_threshold_type="standard_deviation",
            breakpoint_threshold_amount=1.0
        )

    def process_pdf(self, pdf_path: str):

        # 1. Load PDF
        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        # 2. Extract text only
        texts = [doc.page_content for doc in docs]

        # 3. Semantic chunking
        chunks = self.chunker.create_documents(texts)

        # 4. Add metadata manually to chunks
        for i, chunk in enumerate(chunks):

            # fallback metadata
            chunk.metadata["source"] = os.path.basename(pdf_path)

            # try mapping page numbers approximately
            if i < len(docs):
                chunk.metadata["page"] = docs[i].metadata.get("page", i + 1)
            else:
                chunk.metadata["page"] = i + 1

        return chunks


# =========================
# MAIN RAG PIPELINE
# =========================

class RAGPipeline:

    def __init__(self):

        print("Initializing pipeline...")

        self.llm0 = llmsetup()

        self.embedder0 = Embedder()

        self.preprocessor = Preprocessor(self.embedder0)

        print("Pipeline initialized successfully")


    def run(self, query: str, pdf_path: str = None):

        # =========================
        # PDF MODE
        # =========================

        if pdf_path:

            # 1. Process PDF
            chunks = self.preprocessor.process_pdf(pdf_path)

            # 2. Create vector DB
            store = FAISS.from_documents(
                chunks,
                self.embedder0.Emodel
            )

            # 3. Retriever
            retriever = store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": 3,
                    "lambda_mult": 1
                }
            )

            # 4. Retrieve docs
            retrieved_docs = retriever.invoke(query)

            # 5. Build context
            context = "\n\n".join([
                doc.page_content for doc in retrieved_docs
            ])

            # 6. Prompt
            prompt = PromptTemplate(
                template="""
                You are a helpful Legal assistant.

                Answer ONLY from the provided PDF context.

                If answer is not present in context,
                say that clearly.

                Context:
                {context}

                Question:
                {question}
                """,

                input_variables=["context", "question"]
            )

            final_prompt = prompt.invoke({
                "context": context,
                "question": query
            })

            # 7. LLM response
            answer = self.llm0(final_prompt)

            # =========================
            # CITATION EXTRACTION
            # =========================

            sources = []

            seen = set()

            for doc in retrieved_docs:

                file_name = doc.metadata.get("source", "Unknown")

                page_number = doc.metadata.get("page", "Unknown")

                unique_key = f"{file_name}_{page_number}"

                if unique_key not in seen:

                    seen.add(unique_key)

                    sources.append({
                        "file": file_name,
                        "page": page_number
                    })

            # =========================
            # FINAL RESPONSE
            # =========================

            return {
                "answer": answer,
                "sources": sources
            }

        # =========================
        # NO PDF MODE
        # =========================

        else:

            warning_prompt = f"""
            No PDF uploaded.

            Answer the question using your general knowledge and available data to you.

            Also mention briefly that no PDF context was provided.

            Question:
            {query}
            """

            answer = self.llm0(warning_prompt)

            return {
                "answer": answer,
                "sources": []
            }