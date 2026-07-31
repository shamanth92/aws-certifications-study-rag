from sqlalchemy.orm import sessionmaker
from models.conversation import engine, Message

SessionLocal = sessionmaker(bind=engine)

def add_message(conversation_id: str, role: str, content: str):
    with SessionLocal() as session:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        session.add(message)
        session.commit()

def get_history(conversation_id: str) -> list[dict]:
    with SessionLocal() as session:
        messages = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id.asc())
            .all()
        )
        return [{"role": m.role, "content": m.content} for m in messages]
