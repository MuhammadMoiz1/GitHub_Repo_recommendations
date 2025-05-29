from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from github import Github
from recommendation import hybrid_recommendations, get_github_client

app = FastAPI()

# ✅ CORS middleware to allow requests from Flutter Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Replace with ["http://localhost:5000"] or your deployed web URL for more security
    allow_credentials=True,
    allow_methods=["*"],  # Allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],  # Allow all headers including Content-Type
)

# ✅ Request model
class TokenRequest(BaseModel):
    token: str

# ✅ POST endpoint
@app.post("/recommend")
def recommend_repos(req: TokenRequest):
    try:
        g = get_github_client(req.token)
        recommendations = hybrid_recommendations(g, req.token)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
