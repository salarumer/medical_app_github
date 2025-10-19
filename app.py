"""
Flask API for Medical Textbook Retrieval - Cloud Version
========================================================
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

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
        
        print(f"🔌 Connecting to Elasticsearch at {es_host}")
        
        # Connect to Elasticsearch
        es_client = Elasticsearch([es_host], request_timeout=60, max_retries=3)
        
        if not es_client.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")
        
        print(f"✅ Connected to Elasticsearch!")
        
        # Load sentence transformer model
        print(f"🤖 Loading embedding model...")
        model = SentenceTransformer("abhinand/MedEmbed-large-v0.1")
        
        print(f"✅ Model loaded!")
        
        # Check index
        count = es_client.count(index="textbook_search")['count']
        print(f"📚 Index contains {count} documents")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
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
        
        print(f"🔍 Searching for: '{query}' (top_k={top_k})")
        
        results = search_elasticsearch(query, top_k)
        
        print(f"✅ Found {len(results)} results")
        
        return jsonify({
            'status': 'success',
            'query': query,
            'num_results': len(results),
            'results': results
        })
    
    except Exception as e:
        print(f"❌ Search error: {e}")
        return jsonify({
            'status': 'error',
            'message': f'Search failed: {str(e)}'
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏥 Medical Textbook Retrieval API")
    print("="*60)
    
    if not initialize_search_engine():
        print("\n❌ Failed to initialize. Exiting.")
        exit(1)
    
    print("\n📍 Running on: http://localhost:5000")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
# """
# Flask API for Medical Textbook Retrieval
# ========================================
# Automatically loads existing chunks from disk
# """

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import os
# from typing import List, Dict, Any

# # Import your existing modules
# from core.elasticsearch_search_engine import ElasticsearchSearchEngine
# from optimized_pipeline import OptimizedTextbookManager

# app = Flask(__name__)
# CORS(app)  # Enable CORS for frontend access

# # Initialize search engine (connects to existing index or builds from chunks)
# print("🚀 Initializing Medical Textbook Retrieval API...")
# search_engine = None

# def initialize_search_engine():
#     """Initialize search engine and load chunks if index is empty"""
#     global search_engine
    
#     try:
#         # Connect to Elasticsearch (using fast lightweight model for testing)
#         search_engine = ElasticsearchSearchEngine(
#             model_name="all-MiniLM-L6-v2",  # Fast 384-dim model
#             es_host="http://localhost:9200",
#             index_name="textbook_search",
#             create_index=False  # Don't rebuild unless needed
#         )
        
#         # Check if index has data
#         stats = search_engine.get_statistics()
        
#         if stats['total_documents'] == 0:
#             print("⚠️  Elasticsearch index is empty!")
#             print("📖 Loading existing chunks from disk...")
            
#             # Load chunks using OptimizedTextbookManager
#             manager = OptimizedTextbookManager()
#             registry = manager.load_registry()
            
#             if len(registry['textbooks']) == 0:
#                 print("❌ No textbooks found in registry!")
#                 print("   Please run optimized_pipeline.py first to process textbooks.")
#                 return False
            
#             print(f"📚 Found {len(registry['textbooks'])} textbooks in registry:")
#             for textbook_id, info in registry['textbooks'].items():
#                 print(f"   • {info['title']}: {info['chunk_count']} chunks")
            
#             # Get collection and rebuild index
#             print("\n🔨 Building Elasticsearch index from existing chunks...")
#             collection = manager.get_collection_for_search()
            
#             # Rebuild with existing chunks (using fast model)
#             search_engine = ElasticsearchSearchEngine(
#                 model_name="all-MiniLM-L6-v2",  # Fast 384-dim model
#                 es_host="http://localhost:9200",
#                 index_name="textbook_search",
#                 create_index=True  # Force rebuild
#             )
#             search_engine.build_index_from_collection(collection, force_rebuild=True)
            
#             # Verify
#             stats = search_engine.get_statistics()
#             print(f"\n✅ Index built successfully!")
#             print(f"📊 Indexed {stats['total_documents']} documents from {stats['total_textbooks']} textbooks")
#         else:
#             print(f"✅ Connected to existing Elasticsearch index")
#             print(f"📚 Index contains {stats['total_documents']} documents from {stats['total_textbooks']} textbooks")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Failed to initialize search engine: {e}")
#         print("Make sure Elasticsearch is running on localhost:9200")
#         return False


# def format_search_results(results: List[Dict]) -> List[Dict[str, Any]]:
#     """Format search results for API response"""
#     formatted_results = []
    
#     for i, result in enumerate(results, 1):
#         metadata = result['metadata']
        
#         formatted_result = {
#             'rank': i,
#             'relevance_score': float(result['score']),
#             'content_preview': result['content'][:300] + "..." if len(result['content']) > 300 else result['content'],
#             'full_content': result['content'],
#             'source': {
#                 'textbook': metadata.get('textbook', 'Unknown'),
#                 'textbook_id': metadata.get('textbook_id', 'unknown'),
#                 'chapter': metadata.get('chapter', 'Unknown'),
#                 'page': metadata.get('page', 'Unknown'),
#                 'chapter_range': f"{metadata.get('chapter_start_page', '?')}-{metadata.get('chapter_end_page', '?')}"
#             },
#             'citation': result['citation']
#         }
        
#         formatted_results.append(formatted_result)
    
#     return formatted_results


# @app.route('/', methods=['GET'])
# def home():
#     """Health check endpoint"""
#     if search_engine is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Search engine not initialized'
#         }), 500
    
#     stats = search_engine.get_statistics()
    
#     return jsonify({
#         'status': 'healthy',
#         'message': 'Medical Textbook Retrieval API',
#         'version': '1.0.0',
#         'statistics': {
#             'total_documents': stats['total_documents'],
#             'total_textbooks': stats['total_textbooks'],
#             'textbooks': stats['textbook_breakdown'],
#             'embedding_model': stats['model_name']
#         }
#     })


# @app.route('/search', methods=['POST'])
# def search():
#     """
#     Search endpoint for retrieving relevant textbook content
    
#     Request JSON:
#     {
#         "query": "What causes tuberculosis?",
#         "top_k": 5  # optional, default 5
#     }
#     """
#     if search_engine is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Search engine not initialized'
#         }), 500
    
#     try:
#         # Get request data
#         data = request.get_json()
        
#         if not data or 'query' not in data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Missing required field: query'
#             }), 400
        
#         query = data['query']
#         top_k = data.get('top_k', 5)
        
#         # Validate inputs
#         if not query or not query.strip():
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Query cannot be empty'
#             }), 400
        
#         if not isinstance(top_k, int) or top_k < 1 or top_k > 20:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'top_k must be an integer between 1 and 20'
#             }), 400
        
#         print(f"🔍 Searching for: '{query}' (top_k={top_k})")
        
#         # Perform search
#         results = search_engine.search(query, top_k=top_k)
        
#         # Format results
#         formatted_results = format_search_results(results)
        
#         print(f"✅ Found {len(formatted_results)} results")
        
#         return jsonify({
#             'status': 'success',
#             'query': query,
#             'num_results': len(formatted_results),
#             'results': formatted_results
#         })
    
#     except Exception as e:
#         print(f"❌ Search error: {e}")
#         return jsonify({
#             'status': 'error',
#             'message': f'Search failed: {str(e)}'
#         }), 500


# @app.route('/search/qa', methods=['POST'])
# def search_qa():
#     """
#     Search endpoint specifically for question-answer pairs
#     Returns sources for a given question and answer
    
#     Request JSON:
#     {
#         "question": "What is tuberculosis caused by?",
#         "answer": "Mycobacterium tuberculosis",  # optional
#         "top_k": 5  # optional, default 5
#     }
#     """
#     if search_engine is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Search engine not initialized'
#         }), 500
    
#     try:
#         data = request.get_json()
        
#         if not data or 'question' not in data:
#             return jsonify({
#                 'status': 'error',
#                 'message': 'Missing required field: question'
#             }), 400
        
#         question = data['question']
#         answer = data.get('answer', '')
#         top_k = data.get('top_k', 5)
        
#         print(f"🔍 Q&A Search - Question: '{question[:100]}...'")
#         if answer:
#             print(f"   Answer: '{answer}'")
        
#         # Search using the question
#         results = search_engine.search(question, top_k=top_k)
#         formatted_results = format_search_results(results)
        
#         # Create response
#         response = {
#             'status': 'success',
#             'question': question,
#             'answer': answer if answer else None,
#             'num_sources': len(formatted_results),
#             'sources': formatted_results,
#             'primary_source': formatted_results[0] if formatted_results else None
#         }
        
#         print(f"✅ Found {len(formatted_results)} relevant sources")
        
#         return jsonify(response)
    
#     except Exception as e:
#         print(f"❌ Q&A search error: {e}")
#         return jsonify({
#             'status': 'error',
#             'message': f'Q&A search failed: {str(e)}'
#         }), 500


# @app.route('/textbooks', methods=['GET'])
# def list_textbooks():
#     """Get list of available textbooks"""
#     if search_engine is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Search engine not initialized'
#         }), 500
    
#     try:
#         textbooks = search_engine.get_textbook_list()
#         stats = search_engine.get_statistics()
        
#         textbook_details = []
#         for textbook_id in textbooks:
#             doc_count = stats['textbook_breakdown'].get(textbook_id, 0)
#             textbook_details.append({
#                 'id': textbook_id,
#                 'document_count': doc_count
#             })
        
#         return jsonify({
#             'status': 'success',
#             'total_textbooks': len(textbooks),
#             'textbooks': textbook_details
#         })
    
#     except Exception as e:
#         print(f"❌ Error listing textbooks: {e}")
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to list textbooks: {str(e)}'
#         }), 500


# @app.route('/stats', methods=['GET'])
# def get_stats():
#     """Get detailed system statistics"""
#     if search_engine is None:
#         return jsonify({
#             'status': 'error',
#             'message': 'Search engine not initialized'
#         }), 500
    
#     try:
#         stats = search_engine.get_statistics()
        
#         return jsonify({
#             'status': 'success',
#             'statistics': stats
#         })
    
#     except Exception as e:
#         print(f"❌ Error getting stats: {e}")
#         return jsonify({
#             'status': 'error',
#             'message': f'Failed to get statistics: {str(e)}'
#         }), 500


# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("🏥 Medical Textbook Retrieval API")
#     print("="*60)
    
#     # Initialize search engine
#     if not initialize_search_engine():
#         print("\n❌ Failed to initialize. Exiting.")
#         exit(1)
    
#     print("\n📍 Running on: http://localhost:5000")
#     print("📚 Using Elasticsearch with auto-loaded chunks")
#     print("\nAvailable endpoints:")
#     print("  GET  /           - Health check")
#     print("  POST /search     - Search textbooks")
#     print("  POST /search/qa  - Search for Q&A pairs")
#     print("  GET  /textbooks  - List available textbooks")
#     print("  GET  /stats      - Get system statistics")
#     print("="*60 + "\n")
    
#     # Run Flask app
#     app.run(
#         host='0.0.0.0',  # Allow external connections
#         port=5000,
#         debug=True  # Enable debug mode for development
#     )