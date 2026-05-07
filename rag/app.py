
from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from database import engine, SessionLocal
from models import Base, PDF, ChatMessage

from rag.pipeline import RAGPipeline

import shutil
import os
import traceback


# =========================
# FASTAPI APP
# =========================

app = FastAPI()

Base.metadata.create_all(bind=engine)


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


# =========================
# CREATE REQUIRED FOLDERS
# =========================

os.makedirs("uploads", exist_ok=True)


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

    db = SessionLocal()

    try:

        current_pdf = None


        # =========================
        # EXISTING PDF SELECTED
        # =========================

        if selected_pdf:

            pdf_path = f"uploads/{selected_pdf}"

            current_pdf = db.query(PDF).filter(
                PDF.pdf_name == selected_pdf
            ).first()


        # =========================
        # NEW PDF UPLOAD
        # =========================

        if pdf:

            os.makedirs("uploads", exist_ok=True)

            pdf_path = f"uploads/{pdf.filename}"

            with open(pdf_path, "wb") as buffer:

                shutil.copyfileobj(pdf.file, buffer)


            # =========================
            # STORE PDF IN DATABASE
            # =========================

            existing_pdf = db.query(PDF).filter(
                PDF.pdf_name == pdf.filename
            ).first()

            if not existing_pdf:

                new_pdf = PDF(
                    pdf_name=pdf.filename
                )

                db.add(new_pdf)

                db.commit()

                current_pdf = new_pdf

            else:

                current_pdf = existing_pdf


        # =========================
        # SAVE USER MESSAGE
        # =========================

        if current_pdf:

            user_message = ChatMessage(

                pdf_id=current_pdf.id,

                sender="user",

                message=query
            )

            db.add(user_message)

            db.commit()


        # =========================
        # RUN RAG PIPELINE
        # =========================

        result = pipeline.run(

            query=query,

            pdf_path=pdf_path
        )


        # =========================
        # SAVE ASSISTANT RESPONSE
        # =========================

        if current_pdf:

            assistant_message = ChatMessage(

                pdf_id=current_pdf.id,

                sender="assistant",

                message=result["answer"]
            )

            db.add(assistant_message)

            db.commit()


        db.close()


        # =========================
        # RETURN RESPONSE
        # =========================

        return {

            "answer": result["answer"],

            "sources": result["sources"]
        }


    except Exception as e:

        db.close()

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
# LOAD CHAT HISTORY
# =========================

@app.get("/chat-history/{pdf_name}")
async def get_chat_history(pdf_name: str):

    db = SessionLocal()

    try:

        pdf_entry = db.query(PDF).filter(
            PDF.pdf_name == pdf_name
        ).first()

        if not pdf_entry:

            db.close()

            return {
                "messages": []
            }

        messages = db.query(ChatMessage).filter(
            ChatMessage.pdf_id == pdf_entry.id
        ).all()

        history = []

        for msg in messages:

            history.append({

                "sender": msg.sender,

                "message": msg.message
            })

        db.close()

        return {
            "messages": history
        }

    except Exception as e:

        db.close()

        return JSONResponse(

            content={
                "error": str(e)
            },

            status_code=500
        )


# =========================
# DELETE PDF
# =========================

@app.delete("/delete-pdf/{filename}")
async def delete_pdf(filename: str):

    db = SessionLocal()

    try:

        file_path = f"uploads/{filename}"


        # =========================
        # DELETE PHYSICAL FILE
        # =========================

        if os.path.exists(file_path):

            os.remove(file_path)


        # =========================
        # DELETE DB PDF ENTRY
        # =========================

        pdf_entry = db.query(PDF).filter(
            PDF.pdf_name == filename
        ).first()


        if pdf_entry:

            # =========================
            # DELETE ALL RELATED CHATS
            # =========================

            db.query(ChatMessage).filter(
                ChatMessage.pdf_id == pdf_entry.id
            ).delete()


            # =========================
            # DELETE PDF ENTRY
            # =========================

            db.delete(pdf_entry)

            db.commit()


        db.close()

        return {
            "message": "PDF deleted successfully"
        }


    except Exception as e:

        db.close()

        return JSONResponse(

            content={
                "error": str(e)
            },

            status_code=500
        )

