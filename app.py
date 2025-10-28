"""
Flask API for Medical Textbook Retrieval - Cloud Version
========================================================
Multi-Category Support: Emergency Medicine + Gynecology
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
import logging
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

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

# Category to index mapping
CATEGORY_INDEXES = {
    'emergency': 'textbook_search',
    'gynecology': 'gynecology_search'
}

def initialize_search_engine():
    """Initialize connection to Elasticsearch and Gemini"""
    global es_client, model, gemini_model
    
    try:
        # Get Elasticsearch host from environment
        es_host = os.environ.get('ELASTICSEARCH_HOST')
        if not es_host:
            raise ValueError("ELASTICSEARCH_HOST environment variable is required")
        
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
        
        # Initialize Gemini
        gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if gemini_api_key:
            logger.info("🔮 Initializing Gemini API...")
            genai.configure(api_key=gemini_api_key)
            gemini_model = genai.GenerativeModel('gemini-2.0-flash-exp')
            logger.info("✅ Gemini initialized!")
        else:
            logger.warning("⚠️  No GEMINI_API_KEY found - text cleaning disabled")
        
        # Check both indexes
        for category, index_name in CATEGORY_INDEXES.items():
            if es_client.indices.exists(index=index_name):
                count = es_client.count(index=index_name)['count']
                logger.info(f"📚 Index '{index_name}' ({category}): {count} documents")
            else:
                logger.warning(f"⚠️  Index '{index_name}' ({category}) does not exist")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}", exc_info=True)
        return False


def clean_text_with_gemini(text: str) -> str:
    """Clean and format text using Gemini API"""
    if not gemini_model:
        return text  # Return original if Gemini not available
    
    try:
        prompt = f"""Clean and format this medical text for better readability. Remove unnecessary newlines, fix spacing, and make it flow naturally as prose. Keep all medical information intact.

Text to clean:
{text}

Return only the cleaned text without any additional commentary."""

        response = gemini_model.generate_content(prompt)
        cleaned = response.text.strip()
        
        logger.info(f"✨ Cleaned text: {len(text)} → {len(cleaned)} chars")
        return cleaned
        
    except Exception as e:
        logger.error(f"❌ Gemini cleaning error: {e}")
        return text  # Return original on error


def search_elasticsearch(query: str, top_k: int = 5, clean_text: bool = True, category: str = 'emergency') -> List[Dict[str, Any]]:
    """Search Elasticsearch for relevant documents"""
    
    # Get index name for category
    index_name = CATEGORY_INDEXES.get(category, 'textbook_search')
    
    logger.info(f"🔍 Searching in index: {index_name} (category: {category})")
    
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
    response = es_client.search(index=index_name, body=es_query)
    
    # Format results
    results = []
    for i, hit in enumerate(response['hits']['hits'], 1):
        source = hit['_source']
        metadata = source['metadata']
        
        # Get content
        original_content = source['content']
        full_content = original_content
        
        # Clean content if requested and Gemini is available
        if clean_text and gemini_model:
            full_content = clean_text_with_gemini(original_content)
        
        # Create preview from cleaned content
        content_preview = full_content[:300] + "..." if len(full_content) > 300 else full_content
        
        result = {
            'rank': i,
            'relevance_score': float(hit['_score']),
            'content_preview': content_preview,
            'full_content': full_content,
            'original_content': original_content,  # Keep original for reference
            'cleaned': clean_text and gemini_model is not None,
            'category': category,  # NEW: Include category
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
        # Get counts for all indexes
        indexes_stats = {}
        for category, index_name in CATEGORY_INDEXES.items():
            if es_client.indices.exists(index=index_name):
                count = es_client.count(index=index_name)['count']
                indexes_stats[category] = {
                    'index': index_name,
                    'documents': count
                }
        
        return jsonify({
            'status': 'healthy',
            'message': 'Medical Textbook Retrieval API',
            'version': '2.0.0',  # Updated version
            'features': {
                'text_cleaning': gemini_model is not None,
                'multi_category': True,
                'categories': list(CATEGORY_INDEXES.keys())
            },
            'statistics': {
                'indexes': indexes_stats,
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
        clean_text = data.get('clean_text', True)
        category = data.get('category', 'emergency')  # NEW: Get category
        
        # Validate category
        if category not in CATEGORY_INDEXES:
            return jsonify({
                'status': 'error',
                'message': f'Invalid category. Must be one of: {list(CATEGORY_INDEXES.keys())}'
            }), 400
        
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
        
        logger.info(f"🔍 Searching for: '{query}' (category={category}, top_k={top_k}, clean={clean_text})")
        
        results = search_elasticsearch(query, top_k, clean_text, category)
        
        logger.info(f"✅ Found {len(results)} results in {category}")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'category': category,
            'num_results': len(results),
            'text_cleaned': gemini_model is not None and clean_text,
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
        clean_text = data.get('clean_text', True)
        category = data.get('category', 'emergency')  # NEW: Get category
        
        # Validate category
        if category not in CATEGORY_INDEXES:
            return jsonify({
                'status': 'error',
                'message': f'Invalid category. Must be one of: {list(CATEGORY_INDEXES.keys())}'
            }), 400
        
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
        
        logger.info(f"🔍 Enhanced search (Q+A+E): category={category}, {len(enhanced_query)} chars, top_k={top_k}, clean={clean_text}")
        
        # Search with enhanced query in selected category
        results = search_elasticsearch(enhanced_query, top_k, clean_text, category)
        
        logger.info(f"✅ Found {len(results)} results in {category}")
        
        return jsonify({
            'status': 'success',
            'category': category,
            'question': question,
            'answer': answer if answer else None,
            'explanation': explanation if explanation else None,
            'query_used': enhanced_query,
            'num_results': len(results),
            'text_cleaned': gemini_model is not None and clean_text,
            'results': results
        })
    
    except Exception as e:
        logger.error(f"❌ Enhanced search error: {e}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'Enhanced search failed: {str(e)}'
        }), 500


@app.route('/categories', methods=['GET'])
def list_categories():
    """Get list of available categories"""
    if es_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        categories_info = []
        for category, index_name in CATEGORY_INDEXES.items():
            if es_client.indices.exists(index=index_name):
                count = es_client.count(index=index_name)['count']
                
                # Get textbook list for this category
                query = {
                    "size": 0,
                    "aggs": {
                        "textbooks": {
                            "terms": {
                                "field": "metadata.textbook",
                                "size": 100
                            }
                        }
                    }
                }
                response = es_client.search(index=index_name, body=query)
                textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
                
                categories_info.append({
                    'category': category,
                    'index': index_name,
                    'documents': count,
                    'textbooks': textbooks,
                    'available': True
                })
            else:
                categories_info.append({
                    'category': category,
                    'index': index_name,
                    'available': False
                })
        
        return jsonify({
            'status': 'success',
            'categories': categories_info
        })
    
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/textbooks', methods=['GET'])
def list_textbooks():
    """Get list of available textbooks (for backward compatibility)"""
    if es_client is None:
        return jsonify({
            'status': 'error',
            'message': 'Search engine not initialized'
        }), 500
    
    try:
        # Default to emergency medicine for backward compatibility
        category = request.args.get('category', 'emergency')
        index_name = CATEGORY_INDEXES.get(category, 'textbook_search')
        
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
        
        response = es_client.search(index=index_name, body=query)
        textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
        
        return jsonify({
            'status': 'success',
            'category': category,
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
        stats_by_category = {}
        total_documents = 0
        
        for category, index_name in CATEGORY_INDEXES.items():
            if es_client.indices.exists(index=index_name):
                count = es_client.count(index=index_name)['count']
                stats_by_category[category] = {
                    'index': index_name,
                    'documents': count
                }
                total_documents += count
        
        return jsonify({
            'status': 'success',
            'statistics': {
                'total_documents': total_documents,
                'categories': stats_by_category,
                'embedding_model': 'abhinand/MedEmbed-large-v0.1',
                'search_engine': 'Elasticsearch',
                'text_cleaning': gemini_model is not None
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
logger.info("🔥 Multi-Category Support: Emergency + Gynecology")
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
