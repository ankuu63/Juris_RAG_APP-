from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import String
from sqlalchemy import Text
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey

from sqlalchemy.orm import relationship

from datetime import datetime

from database import Base


# =========================
# PDF TABLE
# =========================

class PDF(Base):

    __tablename__ = "pdfs"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    pdf_name = Column(
        String,
        unique=True,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    messages = relationship(
        "ChatMessage",
        back_populates="pdf",
        cascade="all, delete"
    )


# =========================
# CHAT MESSAGE TABLE
# =========================

class ChatMessage(Base):

    __tablename__ = "chat_messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    pdf_id = Column(
        Integer,
        ForeignKey("pdfs.id")
    )

    sender = Column(
        String,
        nullable=False
    )

    message = Column(
        Text,
        nullable=False
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )

    pdf = relationship(
        "PDF",
        back_populates="messages"
    )