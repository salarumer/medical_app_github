"""
Flask API for Medical Textbook Retrieval - Enhanced with Gemini Text Cleaning
==============================================================================
Replace your existing app.py with this file
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from google import generativeai as genai

# Configure logging for Cloud Run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global variables
search_engine = None
es_client = None
model = None
gemini_model = None

def initialize_search_engine():
    """Initialize connection to Elasticsearch and Gemini"""
    global es_client, model, gemini_model
    
    try:
        # Get Elasticsearch host from environment or default
        es_host = os.environ.get('ELASTICSEARCH_HOST', 'http://35.225.123.242:9200')
        
        logger.info(f"🔌 Connecting to Elasticsearch at {es_host}")
        
        # Connect to Elasticsearch
        es_client = Elasticsearch([es_host], request_timeout=60, max_retries=3)
        
        if not es_client.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")
        
        logger.info("✅ Connected to Elasticsearch!")
        
        # Set HuggingFace token if available
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("🔑 Using HuggingFace token for authenticated download")
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        else:
            logger.warning("⚠️  No HF_TOKEN found - may hit rate limits")
        
        # Load sentence transformer model
        logger.info("🤖 Loading embedding model... This may take 2-3 minutes...")
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        
        logger.info("✅ Embedding model loaded successfully!")
        
        # Initialize Gemini for text cleaning
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if gemini_api_key:
            logger.info("🤖 Initializing Gemini 2.5 Pro for text cleaning...")
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel('gemini-2.5-pro')
            logger.info("✅ Gemini initialized successfully!")
        else:
            logger.warning("⚠️  No GEMINI_API_KEY found - text cleaning disabled")
        
        # Check index
        count = es_client.count(index="textbook_search")['count']
        logger.info(f"📚 Index contains {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}", exc_info=True)
        return False


def clean_text_with_gemini(raw_text: str, textbook_name: str, chapter: str) -> Dict[str, str]:
    """
    Clean and format retrieved textbook chunks using Gemini
    
    Args:
        raw_text: Raw text from Elasticsearch
        textbook_name: Name of the textbook
        chapter: Chapter name
    
    Returns:
        Dict with cleaned_text and preview
    """
    if not gemini_model:
        # If Gemini not available, return original text
        return {
            'cleaned_text': raw_text,
            'preview': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
        }
    
    try:
        # Construct prompt for Gemini
        prompt = f"""You are a medical text formatting assistant. Your task is to clean and format a chunk of text extracted from a medical textbook.

**Source Information:**
- Textbook: {textbook_name}
- Chapter: {chapter}

**Raw Text Chunk:**
{raw_text}

**Your Task:**
1. If the text starts with an incomplete sentence fragment, DELETE it completely - start from the first COMPLETE sentence
2. If the text ends with an incomplete sentence fragment, DELETE it completely - end at the last COMPLETE sentence
3. Replace "\\n" with actual line breaks for readability
4. Remove random page numbers, headers, footers that interrupt the text
5. Fix obvious OCR errors (like "gl aucoma" → "glaucoma")
6. Preserve ALL medical terminology, drug names, dosages, measurements, and clinical information EXACTLY as written

**CRITICAL RULES:**
- DO NOT add any text that isn't in the original
- DO NOT try to complete incomplete sentences
- DO NOT infer or guess what came before or after
- DO NOT summarize or paraphrase
- DO NOT remove any complete sentences or clinical information
- DO NOT change drug dosages, measurements, or medical terms
- ONLY remove incomplete sentence fragments at the start/end
- ONLY fix obvious formatting issues

**Example:**
Input: "historical details to the treating physician.\\nPhysical Examination. The pain in acute mesenteric ischemia is\\ntypically described as being out of proportion..."

Output: "Physical Examination. The pain in acute mesenteric ischemia is typically described as being out of proportion..."

(Notice: removed incomplete fragment "historical details to the treating physician." and cleaned \\n)

Return ONLY the cleaned text, no explanations or additional commentary.
"""

        # Call Gemini to clean the text
        response = gemini_model.generate_content(
            prompt,
            generation_config={
                'temperature': 0.1,  # Low temperature for consistent cleaning
                'max_output_tokens': 2048,
            }
        )
        
        cleaned_text = response.text.strip()
        
        # Create a preview (first 300 chars)
        preview = cleaned_text[:300] + "..." if len(cleaned_text) > 300 else cleaned_text
        
        logger.info(f"✅ Text cleaned with Gemini (original: {len(raw_text)} chars → cleaned: {len(cleaned_text)} chars)")
        
        return {
            'cleaned_text': cleaned_text,
            'preview': preview
        }
        
    except Exception as e:
        logger.error(f"❌ Error cleaning text with Gemini: {e}")
        # Fallback to original text if Gemini fails
        return {
            'cleaned_text': raw_text,
            'preview': raw_text[:300] + "..." if len(raw_text) > 300 else raw_text
        }


def search_elasticsearch(query: str, top_k: int = 5, clean_with_gemini: bool = True) -> List[Dict[str, Any]]:
    """Search Elasticsearch for relevant documents and optionally clean with Gemini"""
    
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Build Elasticsearch query
    es_query = {
        "size": top_k,
        "query": {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "Math.max(cosineSimilarity(params.query_vector, 'embedding') + 1.0, 0.001)",
                    "params": {"query_vector": query_embedding.tolist()}
                }
            }
        }
    }
    
    # Execute search
    response = es_client.search(index="textbook_search", body=es_query)
    
    # Format results
    results = []
    for i, hit in enumerate(response['hits']['hits'], 1):
        source = hit['_source']
        metadata = source['metadata']
        raw_content = source['content']
        
        # Clean text with Gemini if enabled
        if clean_with_gemini:
            cleaned = clean_text_with_gemini(
                raw_content, 
                metadata.get('textbook', 'Unknown'),
                metadata.get('chapter', 'Unknown')
            )
            content = cleaned['cleaned_text']
            preview = cleaned['preview']
        else:
            content = raw_content
            preview = raw_content[:300] + "..." if len(raw_content) > 300 else raw_content
        
        result = {
            'rank': i,
            'relevance_score': float(hit['_score']),
            'content_preview': preview,
            'full_content': content,
            'raw_content': raw_content,  # Include raw for reference
            'source': {
                'textbook': metadata.get('textbook', 'Unknown'),
                'textbook_id': metadata.get('textbook_id', 'unknown'),
                'chapter': metadata.get('chapter', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'chapter_range': f"{metadata.get('chapter_start_page', '?')}-{metadata.get('chapter_end_page', '?')}"
            },
            'citation': f"{metadata.get('textbook', 'Unknown')}, {metadata.get('chapter', 'Unknown')}, page {metadata.get('page', 'Unknown')}",
            'cleaned_with_gemini': clean_with_gemini and gemini_model is not None
        }
        results.append(result)
    
    return results


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    if es_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        count = es_client.count(index="textbook_search")['count']
        
        return jsonify({
            'status': 'healthy',
            'message': 'Medical Textbook Retrieval API - Enhanced with Gemini',
            'version': '2.0.0',
            'statistics': {
                'total_documents': count,
                'embedding_model': 'abhinand/MedEmbed-large-v0.1',
                'text_cleaning': 'Gemini 2.5 Pro' if gemini_model else 'Disabled'
            }
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/search', methods=['POST'])
def search():
    """Search endpoint with optional Gemini text cleaning"""
    if es_client is None or model is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: query'
            }), 400
        
        query = data['query']
        top_k = data.get('top_k', 5)
        clean_text = data.get('clean_text', True)  # Default to cleaning
        
        if not query or not query.strip():
            return jsonify({
                'status': 'error',
                'message': 'Query cannot be empty'
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            return jsonify({
                'status': 'error',
                'message': 'top_k must be an integer between 1 and 20'
            }), 400
        
        logger.info(f"🔍 Searching for: '{query}' (top_k={top_k}, clean_text={clean_text})")
        
        results = search_elasticsearch(query, top_k, clean_with_gemini=clean_text)
        
        logger.info(f"✅ Found {len(results)} results")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'num_results': len(results),
            'results': results,
            'text_cleaned': clean_text and gemini_model is not None
        })
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
        }), 500


@app.route('/search/enhanced', methods=['POST'])
def search_enhanced():
    """Enhanced search endpoint - Question + Answer + Explanation with Gemini cleaning"""
    if es_client is None or model is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing required field: question'
            }), 400
        
        question = data['question']
        answer = data.get('answer', '')
        explanation = data.get('explanation', '')
        top_k = data.get('top_k', 5)
        clean_text = data.get('clean_text', True)  # Default to cleaning
        
        if not question or not question.strip():
            return jsonify({
                'status': 'error',
                'message': 'Question cannot be empty'
            }), 400
        
        if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
            return jsonify({
                'status': 'error',
                'message': 'top_k must be an integer between 1 and 20'
            }), 400
        
        # Construct enhanced query (Q+A+E)
        query_parts = [question]
        
        if answer and answer.strip():
            query_parts.append(f"Answer: {answer}")
        
        if explanation and explanation.strip():
            clean_explanation = explanation.replace("Correct Answer:", "").strip()
            if clean_explanation:
                query_parts.append(clean_explanation[:200])
        
        enhanced_query = " ".join(query_parts)
        
        logger.info(f"🔍 Enhanced search (Q+A+E): {len(enhanced_query)} chars, top_k={top_k}, clean_text={clean_text}")
        
        # Search with enhanced query
        results = search_elasticsearch(enhanced_query, top_k, clean_with_gemini=clean_text)
        
        logger.info(f"✅ Found {len(results)} results")
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer if answer else None,
            'explanation': explanation if explanation else None,
            'query_used': enhanced_query,
            'num_results': len(results),
            'results': results,
            'text_cleaned': clean_text and gemini_model is not None
        })
    
    except Exception as e:
        logger.error(f"❌ Enhanced search error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Enhanced search failed: {str(e)}'
        }), 500


@app.route('/textbooks', methods=['GET'])
def list_textbooks():
    """Get list of available textbooks"""
    if es_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        query = {
            "size": 0,
            "aggs": {
                "textbooks": {
                    "terms": {
                        "field": "metadata.textbook_id",
                        "size": 1000
                    }
                }
            }
        }
        
        response = es_client.search(index="textbook_search", body=query)
        textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
        
        return jsonify({
            'status': 'success',
            'total_textbooks': len(textbooks),
            'textbooks': textbooks
        })
    
    except Exception as e:
        logger.error(f"Error listing textbooks: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    if es_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        count = es_client.count(index="textbook_search")['count']
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'total_documents': count,
                'embedding_model': 'abhinand/MedEmbed-large-v0.1',
                'search_engine': 'Elasticsearch',
                'text_cleaning': 'Gemini 2.5 Pro' if gemini_model else 'Disabled'
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# Initialize on startup (for gunicorn with --preload)
logger.info("="*60)
logger.info("🏥 Medical Textbook Retrieval API - Enhanced with Gemini")
logger.info("="*60)

if not initialize_search_engine():
    logger.error("❌ Failed to initialize search engine")
else:
    logger.info("✅ Search engine initialized successfully")
    logger.info("📍 API is ready to serve requests")


if __name__ == '__main__':
    # This runs only for local development
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Running on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
