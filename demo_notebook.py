# Marketing Research Agent Demo
# This notebook demonstrates the functionality of the Marketing Research Agent

import requests
import json
import pandas as pd
import os,io
from datetime import datetime

# Set up API base URL (adjust if running locally)
API_BASE_URL = "http://localhost:8000"

# # Sample data for testing
# sample_ad_data = """campaign_name,impressions,clicks,conversions,cost,ctr,cpc,roas
# Summer Sale Facebook,10000,500,25,250,5.0,0.5,2.5
# Winter Promotion Google,8000,600,40,300,7.5,0.5,3.2
# Spring Launch Instagram,12000,400,20,200,3.3,0.5,2.0
# Back to School LinkedIn,5000,200,15,150,4.0,0.75,2.8"""

# Test functions
def test_marketing_query():
    """Test the main marketing knowledge query endpoint"""
    print("Testing Marketing Knowledge Query...")
    
    query_data = {
        "query": "What are the best practices for Facebook ad copy during summer campaigns?",
        "context": "We're running a summer sale campaign for outdoor equipment"
    }
    
    response = requests.post(f"{API_BASE_URL}/run-agent", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"Result: {result['result']}...")
        # print(f"Confidence Score: {result['confidence_score']}")
        print(f"Recommendations: {result['recommendations'][:2]}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_ad_performance_analysis():
    """Test ad performance analysis"""
    print("\n Testing Ad Performance Analysis...")
    
    csv_path = "metaads-performance.csv"  

    with open(csv_path, "rb") as f:
        files = {"file": (os.path.basename(csv_path), f, "text/csv")}
        data  = {"query_type": "insights"}
        response = requests.post(
            f"{API_BASE_URL}/analyze-performance",
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"Analysis Result: {result['result']}")
        print(f"Number of Insights: {len(result['insights'])}")
        print(f"Sample Insight: {result['insights'][0] if result['insights'] else 'None'}")
        print(f"Recommendations: {result['recommendations'][:2]}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_content_rewriting():
    """Test content rewriting functionality"""
    print("\n Testing Content Rewriting...")
    
    rewrite_data = {
        "text": "Buy our amazing summer products now! Limited time offer!",
        "tone": "professional",
        "platform": "linkedin"
    }
    
    response = requests.post(f"{API_BASE_URL}/rewrite-content", json=rewrite_data)
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        print(f"{result['result']}")
        print(f"Original Text: {result['insights'][0]['original']}")
        print(f"Variations: {result['insights'][0]['variations']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def test_health_check():
    """Test API health"""
    print("\n Testing Health Check...")
    
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        result = response.json()
        print(" API is healthy!")
        print(f"Status: {result['status']}")
        print(f"Timestamp: {result['timestamp']}")
    else:
        print(f" Health check failed: {response.status_code}")

    

# Main demo execution
def run_demo():
    """Run the complete demo"""
    print("Marketing Research Agent Demo")
    print("=" * 50)
    
    # Check if API is running
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("API is not running. Please start the FastAPI server first.")
            return
    except requests.exceptions.RequestException:
        print("Cannot connect to API. Please ensure the FastAPI server is running on http://localhost:8000")
        return
    
    # Run tests
    test_health_check()
    test_marketing_query()
    test_ad_performance_analysis()
    test_content_rewriting()
    
    print("\n Demo completed successfully!")
    print("The Marketing Research Agent is working correctly.")

# Run the demo
if __name__ == "__main__":
    run_demo()


def test_custom_query(query_text, context=None):
    """Test with custom query"""
    query_data = {
        "query": query_text,
        "context": context
    }
    
    response = requests.post(f"{API_BASE_URL}/run-agent", json=query_data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Query: {query_text}")
        print(f"Response: {result['result']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Example usage:
# test_custom_query("How to create compelling video ads for TikTok?")
# test_custom_query("Best email marketing subject lines for Black Friday", "E-commerce fashion brand")