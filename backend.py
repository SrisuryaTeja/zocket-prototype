from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import io
from datetime import datetime
import uvicorn
from langchain_together import ChatTogether
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from fastapi import UploadFile, File
import chromadb
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Marketing Research Agent", version="1.0.0")


class ContentRewriteRequest(BaseModel):
    text: str
    tone: str = "professional"  # "fun", "professional", "casual", "urgent"
    platform: str = "general"  # "facebook", "google", "instagram", "linkedin"

class MarketingQuery(BaseModel):
    query: str
    context: Optional[str] = None

class AgentResponse(BaseModel):
    result: str
    insights: List[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]

# Global variables for agent components
llm = None
embeddings = None
vector_store = None
memory = None

# Initialize agent components
def initialize_agent():
    global llm, embeddings, vector_store, memory
    
 
    llm = ChatTogether(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0.7,
        max_tokens=1000,
        together_api_key=os.getenv("TOGETHER_API_KEY")
    )


    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize memory for conversation history
    memory = ConversationBufferMemory(return_messages=True)
    
    # Initialize vector store with sample marketing knowledge
    initialize_marketing_knowledge_base()

def initialize_marketing_knowledge_base():
    global vector_store, embeddings
    
    # Sample marketing knowledge base
    marketing_docs = [
        "Facebook ads perform best with visual content and emotional appeals. Use bright colors and clear CTAs.",
        "Google Ads should focus on keyword relevance and landing page quality. Include specific benefits in headlines.",
        "Instagram campaigns work well with user-generated content and influencer partnerships.",
        "LinkedIn ads should maintain professional tone and focus on business value propositions.",
        "Summer sale campaigns should emphasize urgency and seasonal benefits like 'beat the heat' messaging.",
        "Ad copy should include power words like 'exclusive', 'limited time', and 'guaranteed'.",
        "Video ads have 3x higher engagement rates than static images across all platforms.",
        "Retargeting campaigns should focus on cart abandoners with personalized product recommendations.",
        "A/B testing should focus on headlines, CTAs, and visual elements for maximum impact.",
        "Mobile-first design is crucial as 80% of social media browsing happens on mobile devices."
    ]
    
    # Create documents
    documents = [Document(page_content=doc, metadata={"source": "marketing_kb"}) for doc in marketing_docs]
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)

def analyze_ad_performance(csv_data: str, query_type: str) -> Dict[str, Any]:
    try:

        df = pd.read_csv(io.StringIO(csv_data))
        insights = []
        recommendations = []
        
        # Analyze key metrics if they exist
        metric_columns = ['Impressions','Reach','Clicks','Conversions','Cost','Revenue','ctr','cpc','roas']
        available_metrics = [col for col in metric_columns if col.lower() in df.columns.str.lower()]
        if 'Revenue' in available_metrics and 'Cost' in available_metrics and 'roas' not in available_metrics:
           df['roas'] = df['Revenue'] / df['Cost']
           available_metrics.append('roas')

        if available_metrics:
            # Calculate performance insights
            for metric in available_metrics:
                col_name = [col for col in df.columns if col.lower() == metric.lower()][0]
                avg_value = df[col_name].mean()
                best_performer = df.loc[df[col_name].idxmax()]
                worst_performer = df.loc[df[col_name].idxmin()]
                
                insights.append({
                    "metric": metric,
                    "average": float(avg_value),
                    "best_campaign": best_performer.to_dict(),
                    "worst_campaign": worst_performer.to_dict()
                })
        
        # Generate AI-powered recommendations
        performance_summary = df.describe().to_string()
        
        prompt = PromptTemplate(
            input_variables=["performance_data", "query_type"],
            template="""
            Analyze the following ad performance data and provide {query_type}:
            
            Performance Data:
            {performance_data}
            
            Provide specific, actionable recommendations for improving ad performance.
            Focus on creative improvements, targeting optimization, and budget allocation.
            """
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        ai_recommendations = chain.run(performance_data=performance_summary, query_type=query_type)
        
        recommendations.extend(ai_recommendations.split('\n'))
        
        return {
            "insights": insights,
            "recommendations": [r.strip() for r in recommendations if r.strip()],
            "summary_stats": df.describe().to_dict(),
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error analyzing ad performance: {str(e)}")

# Content Rewrite Agent
def rewrite_content(text: str, tone: str, platform: str) -> Dict[str, Any]:
    tone_guidelines = {
        "professional": "formal, authoritative, trustworthy language",
        "fun": "playful, energetic, casual language with emojis",
        "casual": "conversational, friendly, approachable tone",
        "urgent": "action-oriented, time-sensitive, compelling language"
    }
    
    platform_guidelines = {
        "facebook": "engaging, social-friendly with clear CTA",
        "google": "keyword-focused, benefit-driven headlines",
        "instagram": "visual-focused, hashtag-friendly, story-telling",
        "linkedin": "professional, business-value focused",
        "general": "versatile, adaptable to multiple platforms"
    }
    
    prompt = PromptTemplate(
        input_variables=["original_text", "tone", "platform", "tone_guide", "platform_guide"],
        template="""
        Rewrite the following ad text using a {tone} tone for {platform}:
        
        Original Text: {original_text}
        
        Tone Guidelines: {tone_guide}
        Platform Guidelines: {platform_guide}
        
        Provide 3 different variations of the rewritten text.
        Each variation should be optimized for the specified platform and tone.
        """
    )
    
    chain = LLMChain(llm=llm, prompt=prompt)
    
    rewritten_content = chain.run(
        original_text=text,
        tone=tone,
        platform=platform,
        tone_guide=tone_guidelines.get(tone, "neutral tone"),
        platform_guide=platform_guidelines.get(platform, "general platform")
    )
    
    variations = [v.strip() for v in rewritten_content.split('\n') if v.strip()]
    
    return {
        "original_text": text,
        "rewritten_variations": variations,
        "tone": tone,
        "platform": platform,
    }

# Marketing Knowledge Query Agent (RAG)
def query_marketing_knowledge(query: str, context: Optional[str] = None) -> Dict[str, Any]:
    global vector_store

    docs = vector_store.similarity_search(query, k=3)
    retrieved_context = "\n".join([doc.page_content for doc in docs])
    full_context = f"{context}\n\n{retrieved_context}" if context else retrieved_context

    prompt = PromptTemplate(
        input_variables=["full_prompt"],
        template="""
        Based on the following marketing knowledge and context, answer the user's query:

        {full_prompt}

        Provide a comprehensive answer with specific recommendations and best practices.
        Include relevant examples where possible.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    full_prompt = f"Context:\n{full_context}\n\nQuery: {query}"
    response = chain.run(full_prompt=full_prompt)

    return {
        "answer": response,
        "relevant_sources": [doc.page_content for doc in docs]
    }


# API Routes
@app.on_event("startup")
async def startup_event():
    initialize_agent()

@app.get("/")
async def root():
    return {"message": "Marketing Research Agent API", "version": "1.0.0"}

@app.post("/run-agent", response_model=AgentResponse)
async def run_agent(request: MarketingQuery):
    """Main agent endpoint that processes marketing queries"""
    try:
        result = query_marketing_knowledge(request.query, request.context)
        
        return AgentResponse(
            result=result["answer"],
            # confidence_score=result["confidence_score"],
            insights=[{"type": "knowledge_retrieval", "sources": result["relevant_sources"]}],
            recommendations=result["answer"].split('\n')[-3:],  # Last 3 lines as recommendations
            metadata={
                "query": request.query,
                "timestamp": datetime.now().isoformat(),
                "agent_type": "marketing_knowledge_rag"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-performance")
async def analyze_performance(file: UploadFile = File(...), query_type: str = "insights"):
    """Analyze ad performance from uploaded CSV file"""
    try:
        content = await file.read()
        decoded = content.decode("utf-8")
        
        # Call core analysis function (same as before)
        result = analyze_ad_performance(decoded, query_type)

        return AgentResponse(
            result="Ad performance analysis completed",
            # confidence_score=result["confidence_score"],
            insights=result["insights"],
            recommendations=result["recommendations"],
            metadata={
                "query_type": query_type,
                "timestamp": datetime.now().isoformat(),
                "agent_type": "ad_performance_analyzer"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rewrite-content")
async def rewrite_ad_content(request: ContentRewriteRequest):
    """Rewrite ad content for different tones and platforms"""
    try:
        result = rewrite_content(request.text, request.tone, request.platform)
        
        return AgentResponse(
            result=f"Content rewritten for {request.tone} tone on {request.platform}",
            insights=[{
                "type": "content_variations",
                "original": result["original_text"],
                "variations": result["rewritten_variations"]
            }],
            recommendations=[f"Use variation {i+1} for {request.platform}" for i in range(len(result["rewritten_variations"]))],
            metadata={
                "tone": request.tone,
                "platform": request.platform,
                "timestamp": datetime.now().isoformat(),
                "agent_type": "content_rewriter"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)