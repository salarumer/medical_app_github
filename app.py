"""
Flask API for Medical Textbook Retrieval - Cloud Version
========================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

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

def initialize_search_engine():
    """Initialize connection to Elasticsearch"""
    global es_client, model
    
    try:
        # Get Elasticsearch host from environment or default
        es_host = os.environ.get('ELASTICSEARCH_HOST', 'http://35.225.123.242:9200')
        
        logger.info(f"🔌 Connecting to Elasticsearch at {es_host}")
        
        # Connect to Elasticsearch
        es_client = Elasticsearch([es_host], request_timeout=60, max_retries=3)
        
        if not es_client.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")
        
        logger.info("✅ Connected to Elasticsearch!")
        
        # Set HuggingFace token if available (to avoid rate limits)
        hf_token = os.environ.get('HF_TOKEN')
        if hf_token:
            logger.info("🔑 Using HuggingFace token for authenticated download")
            os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        else:
            logger.warning("⚠️  No HF_TOKEN found - may hit rate limits")
        
        # Load sentence transformer model
        logger.info("🤖 Loading embedding model... This may take 2-3 minutes...")
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        
        logger.info("✅ Model loaded successfully!")
        
        # Check index
        count = es_client.count(index="textbook_search")['count']
        logger.info(f"📚 Index contains {count} documents")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}", exc_info=True)
        return False


def search_elasticsearch(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search Elasticsearch for relevant documents"""
    
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
        
        result = {
            'rank': i,
            'relevance_score': float(hit['_score']),
            'content_preview': source['content'][:300] + "..." if len(source['content']) > 300 else source['content'],
            'full_content': source['content'],
            'source': {
                'textbook': metadata.get('textbook', 'Unknown'),
                'textbook_id': metadata.get('textbook_id', 'unknown'),
                'chapter': metadata.get('chapter', 'Unknown'),
                'page': metadata.get('page', 'Unknown'),
                'chapter_range': f"{metadata.get('chapter_start_page', '?')}-{metadata.get('chapter_end_page', '?')}"
            },
            'citation': f"{metadata.get('textbook', 'Unknown')}, {metadata.get('chapter', 'Unknown')}, page {metadata.get('page', 'Unknown')}"
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
            'message': 'Medical Textbook Retrieval API',
            'version': '1.0.0',
            'statistics': {
                'total_documents': count,
                'embedding_model': 'abhinand/MedEmbed-large-v0.1'
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
    """Search endpoint"""
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
        
        logger.info(f"🔍 Searching for: '{query}' (top_k={top_k})")
        
        results = search_elasticsearch(query, top_k)
        
        logger.info(f"✅ Found {len(results)} results")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'num_results': len(results),
            'results': results
        })
    
    except Exception as e:
        logger.error(f"❌ Search error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
        }), 500


@app.route('/search/enhanced', methods=['POST'])
def search_enhanced():
    """Enhanced search endpoint - Question + Answer + Explanation"""
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
        
        logger.info(f"🔍 Enhanced search (Q+A+E): {len(enhanced_query)} chars, top_k={top_k}")
        
        # Search with enhanced query
        results = search_elasticsearch(enhanced_query, top_k)
        
        logger.info(f"✅ Found {len(results)} results")
        
        return jsonify({
            'status': 'success',
            'question': question,
            'answer': answer if answer else None,
            'explanation': explanation if explanation else None,
            'query_used': enhanced_query,
            'num_results': len(results),
            'results': results
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
                'search_engine': 'Elasticsearch'
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
logger.info("🏥 Medical Textbook Retrieval API - Starting")
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
