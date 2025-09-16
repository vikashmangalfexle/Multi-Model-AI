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
class Prompt(Base):
    __tablename__ = "ai_prompts"
    id = Column(Integer, primary_key=True)
    prompt = Column(Text, nullable=False)
    domain = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    responses = relationship("Response", back_populates="prompt", cascade="all, delete")

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
    allow_origins=origins,  # TODO: restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ HELPERS ------------------
def detect_domain_gemini(prompt: str) -> str:
    """
    Use Gemini API to classify the domain of the prompt.
    """
    try:
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        headers = {"X-goog-api-key": os.getenv("GEMINI_API_KEY")}
        body = {
            "contents": [{
                "parts": [{
                    "text": (
                        "Classify the following text into one of these domains: "
                        "programming, finance, health, education, creative, general.\n\n"
                        f"Text: {prompt}\n\n"
                        "Return only the domain name."
                    )
                }]
            }]
        }
        r = requests.post(url, headers=headers, json=body, timeout=15)
        r.raise_for_status()
        domain = r.json()["candidates"][0]["content"]["parts"][0]["text"].strip().lower()
        valid_domains = ["programming", "finance", "health", "education", "creative", "general"]
        return domain if domain in valid_domains else "general"
    except Exception as e:
        print(f"[WARN] Gemini domain detection failed: {e}")
        return "general"

def get_top_providers(db, domain: str, threshold=4):
    """
    Fetch providers with avg_rating >= threshold for a given domain.
    """
    stats = db.query(ProviderStats).filter(ProviderStats.domain == domain).all()
    return [s.provider for s in stats if s.avg_rating >= threshold]

def query_ai_models(prompt: str, providers: list[str]) -> dict:
    """
    Query selected AI providers for responses.
    """
    responses = {}
    for p in providers:
        try:
            if p == "openai":
                r = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}
                )
                r.raise_for_status()
                responses[p] = r.json()["choices"][0]["message"]["content"]

            elif p == "gemini":
                r = requests.post(
                    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                    headers={"X-goog-api-key": os.getenv("GEMINI_API_KEY")},
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                r.raise_for_status()
                responses[p] = r.json()["candidates"][0]["content"]["parts"][0]["text"]

            elif p == "grok":
                r = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {os.getenv('XAI_API_KEY')}"},
                    json={"model": "grok-1", "messages": [{"role": "user", "content": prompt}]}
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
    optimized: bool = True

class FeedbackRequest(BaseModel):
    prompt_id: int
    provider: str
    rating: int

# ------------------ ENDPOINTS ------------------
@app.post("/generate")
def generate(data: PromptRequest):
    db = SessionLocal()
    try:
        # Detect domain for this prompt
        domain = detect_domain_gemini(data.prompt)

        # Get top providers if domain has high rated ones
        top_providers = get_top_providers(db, domain)
        all_providers = ["openai", "gemini", "grok"]

        if data.optimized and top_providers:  # Existing high-rated provider(s)
            providers_to_use = top_providers
            mode = "rated"
        else:  # New domain or no good ratings yet → show all
            providers_to_use = all_providers
            mode = "explore"

        # Query AI providers
        ai_responses = query_ai_models(data.prompt, providers_to_use)

        # Save prompt
        new_prompt = Prompt(prompt=data.prompt, domain=domain)
        db.add(new_prompt)
        db.commit()
        db.refresh(new_prompt)

        # Save responses
        for provider, content in ai_responses.items():
            db.add(Response(prompt_id=new_prompt.id, provider=provider, content=content))
        db.commit()

        return {
            "prompt_id": new_prompt.id,
            "domain": domain,
            "mode": mode,  # "explore" (all providers) or "rated" (top providers only)
            "providers_used": providers_to_use,
            "responses": ai_responses
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

        # Save feedback
        resp.rating = data.rating
        db.commit()

        # Update provider stats
        prompt = resp.prompt
        stat = db.query(ProviderStats).filter(
            ProviderStats.domain == prompt.domain,
            ProviderStats.provider == data.provider
        ).first()

        if not stat:
            stat = ProviderStats(
                domain=prompt.domain,
                provider=data.provider,
                avg_rating=data.rating,
                rating_count=1
            )
            db.add(stat)
        else:
            total = stat.avg_rating * stat.rating_count
            stat.rating_count += 1
            stat.avg_rating = (total + data.rating) / stat.rating_count

        db.commit()
        return {"message": "Feedback saved and stats updated"}
    finally:
        db.close()

