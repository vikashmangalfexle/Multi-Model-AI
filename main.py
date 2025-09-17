import os
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime, Float
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from dotenv import load_dotenv

# ------------------ ENV CONFIG ------------------
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:password@localhost:5432/yourdb")
origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ------------------ DB MODELS ------------------
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=True)
    domain = Column(String(50), nullable=True)   # ✅ persist domain
    created_at = Column(DateTime, default=datetime.utcnow)

    prompts = relationship("Prompt", back_populates="conversation", cascade="all, delete")


class Prompt(Base):
    __tablename__ = "ai_prompts"
    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    prompt = Column(Text, nullable=False)
    domain = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    responses = relationship("Response", back_populates="prompt", cascade="all, delete")
    conversation = relationship("Conversation", back_populates="prompts")


class Response(Base):
    __tablename__ = "ai_responses"
    id = Column(Integer, primary_key=True)
    prompt_id = Column(Integer, ForeignKey("ai_prompts.id", ondelete="CASCADE"))
    provider = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    rating = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

    prompt = relationship("Prompt", back_populates="responses")


class ProviderStats(Base):
    __tablename__ = "provider_stats"
    id = Column(Integer, primary_key=True)
    domain = Column(String(50), nullable=False)
    provider = Column(String(50), nullable=False)
    avg_rating = Column(Float, default=0)
    rating_count = Column(Integer, default=0)


Base.metadata.create_all(bind=engine)

# ------------------ FASTAPI APP ------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ HELPERS ------------------
def detect_domain_gemini(prompt: str) -> str:
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"X-goog-api-key": os.getenv("GEMINI_API_KEY")}
        body = {
            "contents": [{
                "parts": [{
                    "text": (
                        "Classify into one of: programming, finance, health, education, creative, general.\n\n"
                        f"Text: {prompt}\n\nReturn only the domain."
                    )
                }]
            }]
        }
        r = requests.post(url, headers=headers, json=body, timeout=15)
        r.raise_for_status()
        domain = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        return domain if domain in ["programming", "finance", "health", "education", "creative", "general"] else "general"
    except Exception as e:
        print(f"[WARN] Gemini domain detection failed: {e}")
        return "general"


def get_top_providers(db, domain: str, threshold=4):
    stats = db.query(ProviderStats).filter(ProviderStats.domain == domain).all()
    return [s.provider for s in stats if s.avg_rating >= threshold]

SYSTEM_PROMPT = """
You are an assistant inside a multi-AI chat app.
If the user asks "what does this app do", "how does this work", "who are you", 
or anything similar, always explain clearly:

- The app connects to multiple AI providers (OpenAI, Gemini, Grok).
- It automatically detects the domain of the query (programming, finance, health, etc.).
- It can optimize by using top-rated providers for that domain.
- Users can rate responses, and the system learns over time.
- Conversations and feedback are stored for personalization.

Answer in a friendly, concise way unless the user asks for more detail.
Don't give app details unless asked.
"""
def query_ai_models_with_history(prompt: str, history: list[dict], providers: list[str]) -> dict:
    responses = {}
    # Always prepend system prompt
    history = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    for p in providers:
        try:
            if p == "openai":
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    json={"model": "gpt-4o-mini", "messages": history + [{"role": "user", "content": prompt}]}
                )
                r.raise_for_status()
                responses[p] = r.json()["choices"][0]["message"]["content"]

            elif p == "gemini":
                full_context = "\n".join([f"{h['role']}: {h['content']}" for h in history]) + f"\nuser: {prompt}"
                r = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={"X-goog-api-key": os.getenv("GEMINI_API_KEY")},
                    json={"contents": [{"parts": [{"text": full_context}]}]}
                )
                r.raise_for_status()
                responses[p] = r.json()["candidates"][0]["content"]["parts"][0]["text"]

            elif p == "grok":
                r = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"},
                    json={"model": "grok-1", "messages": history + [{"role": "user", "content": prompt}]}
                )
                r.raise_for_status()
                responses[p] = r.json()["choices"][0]["message"]["content"]

            else:
                responses[p] = "Provider not implemented"
        except Exception as e:
            responses[p] = f"Error: {e}"
    return responses

# ------------------ REQUEST SCHEMAS ------------------
class PromptRequest(BaseModel):
    prompt: str
    optimized: bool = False
    conversation_id: int | None = None


class FeedbackRequest(BaseModel):
    prompt_id: int
    provider: str
    rating: int

# ------------------ ENDPOINTS ------------------
@app.post("/generate")
def generate(data: PromptRequest):
    db = SessionLocal()
    try:
        # If no conversation -> new one
        if not data.conversation_id:
            conversation = Conversation(title=data.prompt[:50])
            db.add(conversation)
            db.commit()
            db.refresh(conversation)
            conversation_id = conversation.id
        else:
            conversation = db.query(Conversation).get(data.conversation_id)
            if not conversation:
                raise HTTPException(404, "Conversation not found")
            conversation_id = conversation.id

        # Build history (last 5 turns)
        history = []
        for p in conversation.prompts[-5:]:
            history.append({"role": "user", "content": p.prompt})
            for r in p.responses:
                history.append({"role": "assistant", "content": r.content})

        # Domain: reuse if set, else detect and store
        if conversation.domain:
            domain = conversation.domain
        else:
            domain = detect_domain_gemini(data.prompt)
            conversation.domain = domain
            db.commit()

        # Providers
        top_providers = get_top_providers(db, domain)
        all_providers = ["openai", "gemini", "grok"]
        providers_to_use = top_providers if data.optimized and top_providers else all_providers
        mode = "rated" if providers_to_use == top_providers else "explore"

        # Query models
        ai_responses = query_ai_models_with_history(data.prompt, history, providers_to_use)

        # Save prompt + responses
        new_prompt = Prompt(prompt=data.prompt, domain=domain, conversation_id=conversation_id)
        db.add(new_prompt)
        db.commit()
        db.refresh(new_prompt)

        for provider, content in ai_responses.items():
            db.add(Response(prompt_id=new_prompt.id, provider=provider, content=content))
        db.commit()

        return {
            "conversation_id": conversation_id,
            "prompt_id": new_prompt.id,
            "domain": domain,
            "mode": mode,
            "providers_used": providers_to_use,
            "responses": ai_responses,
        }
    finally:
        db.close()


@app.post("/feedback")
def feedback(data: FeedbackRequest):
    db = SessionLocal()
    try:
        resp = db.query(Response).filter(
            Response.prompt_id == data.prompt_id,
            Response.provider == data.provider
        ).first()
        if not resp:
            raise HTTPException(status_code=404, detail="Response not found")

        resp.rating = data.rating
        db.commit()

        prompt = resp.prompt
        stat = db.query(ProviderStats).filter(
            ProviderStats.domain == prompt.domain,
            ProviderStats.provider == data.provider
        ).first()

        if not stat:
            stat = ProviderStats(domain=prompt.domain, provider=data.provider, avg_rating=data.rating, rating_count=1)
            db.add(stat)
        else:
            total = stat.avg_rating * stat.rating_count
            stat.rating_count += 1
            stat.avg_rating = (total + data.rating) / stat.rating_count

        db.commit()
        return {"message": "Feedback saved and stats updated"}
    finally:
        db.close()


@app.post("/new_conversation")
def new_conversation():
    db = SessionLocal()
    try:
        convo = Conversation(title="New Chat", domain=None)
        db.add(convo)
        db.commit()
        db.refresh(convo)
        return {"conversation_id": convo.id}
    finally:
        db.close()
