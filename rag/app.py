
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from rag.pipeline import RAGPipeline

import shutil
import os
import traceback


# =========================
# FASTAPI APP
# =========================

app = FastAPI()


# =========================
# STATIC FILES
# =========================

app.mount(
    "/static",
    StaticFiles(directory="frontend"),
    name="static"
)


# =========================
# INITIALIZE PIPELINE
# =========================

pipeline = RAGPipeline()

#create required folders
os.makedirs("uploads,exist_ok=True")


# =========================
# HOME PAGE
# =========================

@app.get("/")
def home():

    return FileResponse("frontend/index.html")


# =========================
# CHAT ENDPOINT
# =========================

@app.post("/chat")
async def chat(

    query: str = Form(...),

    pdf: UploadFile = File(None),

    selected_pdf: str = Form(None)
):

    pdf_path = None

    try:

        # =========================
        # EXISTING PDF SELECTED
        # =========================

        if selected_pdf:

            pdf_path = f"uploads/{selected_pdf}"


        # =========================
        # NEW PDF UPLOAD
        # =========================

        if pdf:

            os.makedirs("uploads", exist_ok=True)

            pdf_path = f"uploads/{pdf.filename}"

            with open(pdf_path, "wb") as buffer:

                shutil.copyfileobj(pdf.file, buffer)


        # =========================
        # RUN RAG PIPELINE
        # =========================

        answer = pipeline.run(

            query=query,

            pdf_path=pdf_path
        )

        return {
            "answer": str(answer)
        }


    except Exception as e:

        print("Error in /chat endpoint")

        traceback.print_exc()

        return JSONResponse(

            content={
                "error": str(e)
            },

            status_code=500
        )


# =========================
# GET ALL PDFs
# =========================

@app.get("/pdfs")
async def get_pdfs():

    os.makedirs("uploads", exist_ok=True)

    pdf_files = []

    for file in os.listdir("uploads"):

        if file.endswith(".pdf"):

            pdf_files.append(file)

    return {
        "pdfs": pdf_files
    }


# =========================
# DELETE PDF
# =========================

@app.delete("/delete-pdf/{filename}")
async def delete_pdf(filename: str):

    file_path = f"uploads/{filename}"

    if not os.path.exists(file_path):

        return JSONResponse(

            content={
                "error": "File not found"
            },

            status_code=404
        )

    os.remove(file_path)

    return {
        "message": "PDF deleted successfully"
    }

