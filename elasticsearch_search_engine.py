# """
# Elasticsearch Search Engine
# ==========================

# Drop-in replacement for FAISS using Elasticsearch for vector similarity search.
# """

# import json
# import time
# import numpy as np
# from typing import List, Dict, Any, Optional
# from langchain.schema import Document

# try:
#     from elasticsearch import Elasticsearch
#     from elasticsearch.helpers import bulk
#     HAS_ELASTICSEARCH = True
# except ImportError:
#     HAS_ELASTICSEARCH = False

# try:
#     from sentence_transformers import SentenceTransformer
#     HAS_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     HAS_SENTENCE_TRANSFORMERS = False

# from .models import TextbookCollection


# class ElasticsearchSearchEngine:
#     """Elasticsearch-based search engine with vector similarity"""

#     def __init__(self,
#                  model_name: str = "all-MiniLM-L6-v2",
#                  es_host: str = "localhost:9200",
#                  index_name: str = "textbook_search",
#                  create_index: bool = False):  # ← added parameter

#         if not HAS_ELASTICSEARCH:
#             raise ImportError("elasticsearch not installed. Run: pip install elasticsearch")

#         if not HAS_SENTENCE_TRANSFORMERS:
#             raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

#         self.model_name = model_name
#         self.index_name = index_name
#         self.embedding_dim = 384

#         # Initialize Elasticsearch client
#         self.es = Elasticsearch([es_host])

#         # Check Elasticsearch connection
#         if not self.es.ping():
#             raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")

#         print(f"Connected to Elasticsearch at {es_host}")

#         # Initialize sentence transformer
#         print(f"Loading model: {model_name}")
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         print(f"Model loaded, embedding dimension: {self.embedding_dim}")

#         # Only create index if explicitly requested OR if it doesn't exist
#         if create_index or not self.es.indices.exists(index=self.index_name):
#             self._create_index_mapping()
#         else:
#             print(f"Using existing index: {self.index_name}")
#             # Verify the index has data
#             count = self.es.count(index=self.index_name)['count']
#             print(f"Index contains {count} documents")

#     def _create_index_mapping(self):
#         """Create Elasticsearch index with proper mapping for vector search"""
#         mapping = {
#             "mappings": {
#                 "properties": {
#                     "content": {
#                         "type": "text",
#                         "analyzer": "standard"
#                     },
#                     "embedding": {
#                         "type": "dense_vector",
#                         "dims": self.embedding_dim,
#                         "index": True,
#                         "similarity": "cosine"
#                     },
#                     "metadata": {
#                         "properties": {
#                             "source": {"type": "keyword"},
#                             "textbook_id": {"type": "keyword"},
#                             "textbook": {"type": "keyword"},
#                             "chapter": {"type": "text"},
#                             "chapter_start_page": {"type": "integer"},
#                             "chapter_end_page": {"type": "integer"},
#                             "page": {"type": "integer"},
#                             "chunk_id": {"type": "keyword"},
#                             "chunk_index": {"type": "integer"}
#                         }
#                     }
#                 }
#             },
#             "settings": {
#                 "number_of_shards": 1,
#                 "number_of_replicas": 0
#             }
#         }

#         # Delete existing index if it exists
#         if self.es.indices.exists(index=self.index_name):
#             print(f"Deleting existing index: {self.index_name}")
#             self.es.indices.delete(index=self.index_name)

#         # Create new index
#         print(f"Creating index: {self.index_name}")
#         self.es.indices.create(index=self.index_name, body=mapping)

#     def build_index_from_collection(self, collection: TextbookCollection, force_rebuild: bool = False):
#         """Build search index from textbook collection"""
#         if force_rebuild:
#             self._create_index_mapping()

#         return self._index_documents(collection.all_documents)

#     def build_index_from_documents(self, documents: List[Document],
#                                    cache_name: str = None, force_rebuild: bool = False):
#         """Build search index from document list"""
#         if force_rebuild:
#             self._create_index_mapping()

#         return self._index_documents(documents)

#     def _index_documents(self, documents: List[Document]) -> bool:
#         """Index documents in Elasticsearch"""
#         print(f"Indexing {len(documents)} documents...")

#         # Generate embeddings
#         print("Generating embeddings...")
#         start_time = time.time()
#         texts = [doc.page_content for doc in documents]
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         embed_time = time.time() - start_time
#         print(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")

#         # Prepare documents for bulk indexing
#         actions = []
#         for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#             action = {
#                 "_index": self.index_name,
#                 "_id": f"doc_{i}",
#                 "_source": {
#                     "content": doc.page_content,
#                     "embedding": embedding.tolist(),
#                     "metadata": doc.metadata
#                 }
#             }
#             actions.append(action)

#         # Bulk index documents
#         print("Bulk indexing documents...")
#         start_time = time.time()
#         bulk(self.es, actions)
#         index_time = time.time() - start_time
#         print(f"Indexed {len(actions)} documents in {index_time:.2f}s")

#         # Refresh index
#         self.es.indices.refresh(index=self.index_name)
#         print("Index refreshed and ready for search!")

#         return True

#     def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Search for relevant documents using vector similarity"""

#         # Generate query embedding
#         query_embedding = self.model.encode([query])[0]

#         # Build Elasticsearch query
#         es_query = {
#             "size": top_k,
#             "query": {
#                 "script_score": {
#                     "query": {"match_all": {}},
#                     "script": {
#                         "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
#                         "params": {"query_vector": query_embedding.tolist()}
#                     }
#                 }
#             }
#         }

#         # Add textbook filter if specified
#         if textbook_filter:
#             es_query["query"]["script_score"]["query"] = {
#                 "term": {"metadata.textbook_id": textbook_filter}
#             }

#         # Execute search
#         response = self.es.search(index=self.index_name, body=es_query)

#         # Process results
#         results = []
#         for hit in response['hits']['hits']:
#             source = hit['_source']

#             # Reconstruct Document object
#             doc = Document(
#                 page_content=source['content'],
#                 metadata=source['metadata']
#             )

#             result = {
#                 'document': doc,
#                 'score': hit['_score'],
#                 'content': source['content'],
#                 'metadata': source['metadata'],
#                 'citation': self._format_citation(doc)
#             }
#             results.append(result)

#         return results

#     def hybrid_search(self, query: str, top_k: int = 5,
#                       vector_weight: float = 0.7, text_weight: float = 0.3,
#                       textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Hybrid search combining vector similarity and text matching"""

#         # Generate query embedding
#         query_embedding = self.model.encode([query])[0]

#         # Build hybrid query
#         es_query = {
#             "size": top_k,
#             "query": {
#                 "bool": {
#                     "should": [
#                         {
#                             "script_score": {
#                                 "query": {"match_all": {}},
#                                 "script": {
#                                     "source": f"cosineSimilarity(params.query_vector, 'embedding') * {vector_weight}",
#                                     "params": {"query_vector": query_embedding.tolist()}
#                                 }
#                             }
#                         },
#                         {
#                             "multi_match": {
#                                 "query": query,
#                                 "fields": ["content", "metadata.chapter"],
#                                 "boost": text_weight
#                             }
#                         }
#                     ]
#                 }
#             }
#         }

#         # Add textbook filter if specified
#         if textbook_filter:
#             es_query["query"]["bool"]["filter"] = [
#                 {"term": {"metadata.textbook_id": textbook_filter}}
#             ]

#         # Execute search
#         response = self.es.search(index=self.index_name, body=es_query)

#         # Process results (same as regular search)
#         results = []
#         for hit in response['hits']['hits']:
#             source = hit['_source']

#             doc = Document(
#                 page_content=source['content'],
#                 metadata=source['metadata']
#             )

#             result = {
#                 'document': doc,
#                 'score': hit['_score'],
#                 'content': source['content'],
#                 'metadata': source['metadata'],
#                 'citation': self._format_citation(doc)
#             }
#             results.append(result)

#         return results

#     def _format_citation(self, doc: Document) -> str:
#         """Format citation for a document"""
#         metadata = doc.metadata
#         textbook = metadata.get('textbook', 'Unknown Textbook')
#         chapter = metadata.get('chapter', 'Unknown Chapter')
#         page = metadata.get('page', 'Unknown')

#         return f"{textbook}, {chapter}, page {page}"

#     def get_textbook_list(self) -> List[str]:
#         """Get list of available textbooks"""
#         query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook_id",
#                         "size": 1000
#                     }
#                 }
#             }
#         }

#         response = self.es.search(index=self.index_name, body=query)
#         textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
#         return sorted(textbooks)

#     def get_statistics(self) -> Dict[str, Any]:
#         """Get search engine statistics"""
#         # Get total document count
#         count_response = self.es.count(index=self.index_name)
#         total_documents = count_response['count']

#         # Get textbook breakdown
#         aggs_query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook",
#                         "size": 1000
#                     }
#                 }
#             }
#         }

#         response = self.es.search(index=self.index_name, body=aggs_query)
#         textbook_counts = {
#             bucket['key']: bucket['doc_count']
#             for bucket in response['aggregations']['textbooks']['buckets']
#         }

#         return {
#             'total_documents': total_documents,
#             'total_textbooks': len(textbook_counts),
#             'textbook_breakdown': textbook_counts,
#             'index_built': True,
#             'embedding_dimension': self.embedding_dim,
#             'search_engine': 'Elasticsearch'
#         }

#     def delete_textbook(self, textbook_id: str) -> bool:
#         """Delete all documents for a specific textbook"""
#         query = {
#             "query": {
#                 "term": {
#                     "metadata.textbook_id": textbook_id
#                 }
#             }
#         }

#         response = self.es.delete_by_query(index=self.index_name, body=query)
#         deleted_count = response['deleted']

#         print(f"Deleted {deleted_count} documents for textbook: {textbook_id}")
#         return deleted_count > 0

# """
# Fixed Elasticsearch Search Engine - Handles Negative Cosine Scores
# ===================================================================
# """

# import json
# import time
# import numpy as np
# from typing import List, Dict, Any, Optional
# from langchain.schema import Document

# try:
#     from elasticsearch import Elasticsearch
#     from elasticsearch.helpers import bulk
#     HAS_ELASTICSEARCH = True
# except ImportError:
#     HAS_ELASTICSEARCH = False

# try:
#     from sentence_transformers import SentenceTransformer
#     HAS_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     HAS_SENTENCE_TRANSFORMERS = False

# from .models import TextbookCollection


# class ElasticsearchSearchEngine:
#     """Elasticsearch-based search engine with vector similarity"""

#     def __init__(self,
#                  model_name: str = "all-MiniLM-L6-v2",
#                  es_host: str = "localhost:9200",
#                  index_name: str = "textbook_search",
#                  create_index: bool = False):

#         if not HAS_ELASTICSEARCH:
#             raise ImportError("elasticsearch not installed. Run: pip install elasticsearch")

#         if not HAS_SENTENCE_TRANSFORMERS:
#             raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")

#         self.model_name = model_name
#         self.index_name = index_name
#         self.embedding_dim = 384

#         # Initialize Elasticsearch client
#         self.es = Elasticsearch([es_host])

#         # Check Elasticsearch connection
#         if not self.es.ping():
#             raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")

#         print(f"Connected to Elasticsearch at {es_host}")

#         # Initialize sentence transformer
#         print(f"Loading model: {model_name}")
#         self.model = SentenceTransformer(model_name)
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         print(f"Model loaded, embedding dimension: {self.embedding_dim}")

#         # Only create index if explicitly requested OR if it doesn't exist
#         if create_index or not self.es.indices.exists(index=self.index_name):
#             self._create_index_mapping()
#         else:
#             print(f"Using existing index: {self.index_name}")
#             # Verify the index has data
#             count = self.es.count(index=self.index_name)['count']
#             print(f"Index contains {count} documents")

#     def _create_index_mapping(self):
#         """Create Elasticsearch index with proper mapping for vector search"""
#         mapping = {
#             "mappings": {
#                 "properties": {
#                     "content": {
#                         "type": "text",
#                         "analyzer": "standard"
#                     },
#                     "embedding": {
#                         "type": "dense_vector",
#                         "dims": self.embedding_dim,
#                         "index": True,
#                         "similarity": "cosine"
#                     },
#                     "metadata": {
#                         "properties": {
#                             "source": {"type": "keyword"},
#                             "textbook_id": {"type": "keyword"},
#                             "textbook": {"type": "keyword"},
#                             "chapter": {"type": "text"},
#                             "chapter_start_page": {"type": "integer"},
#                             "chapter_end_page": {"type": "integer"},
#                             "page": {"type": "integer"},
#                             "chunk_id": {"type": "keyword"},
#                             "chunk_index": {"type": "integer"}
#                         }
#                     }
#                 }
#             },
#             "settings": {
#                 "number_of_shards": 1,
#                 "number_of_replicas": 0
#             }
#         }

#         # Delete existing index if it exists
#         if self.es.indices.exists(index=self.index_name):
#             print(f"Deleting existing index: {self.index_name}")
#             self.es.indices.delete(index=self.index_name)

#         # Create new index
#         print(f"Creating index: {self.index_name}")
#         self.es.indices.create(index=self.index_name, body=mapping)

#     def build_index_from_collection(self, collection: TextbookCollection, force_rebuild: bool = False):
#         """Build search index from textbook collection"""
#         if force_rebuild:
#             self._create_index_mapping()

#         return self._index_documents(collection.all_documents)

#     def build_index_from_documents(self, documents: List[Document],
#                                    cache_name: str = None, force_rebuild: bool = False):
#         """Build search index from document list"""
#         if force_rebuild:
#             self._create_index_mapping()

#         return self._index_documents(documents)

#     def _index_documents(self, documents: List[Document]) -> bool:
#         """Index documents in Elasticsearch"""
#         print(f"Indexing {len(documents)} documents...")

#         # Generate embeddings
#         print("Generating embeddings...")
#         start_time = time.time()
#         texts = [doc.page_content for doc in documents]
#         embeddings = self.model.encode(texts, show_progress_bar=True)
#         embed_time = time.time() - start_time
#         print(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")

#         # Prepare documents for bulk indexing
#         actions = []
#         for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#             action = {
#                 "_index": self.index_name,
#                 "_id": f"doc_{i}",
#                 "_source": {
#                     "content": doc.page_content,
#                     "embedding": embedding.tolist(),
#                     "metadata": doc.metadata
#                 }
#             }
#             actions.append(action)

#         # Bulk index documents
#         print("Bulk indexing documents...")
#         start_time = time.time()
#         bulk(self.es, actions)
#         index_time = time.time() - start_time
#         print(f"Indexed {len(actions)} documents in {index_time:.2f}s")

#         # Refresh index
#         self.es.indices.refresh(index=self.index_name)
#         print("Index refreshed and ready for search!")

#         return True

#     def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Search for relevant documents using vector similarity"""

#         # Generate query embedding
#         query_embedding = self.model.encode([query])[0]

#         # FIXED: Use max() to ensure non-negative scores
#         es_query = {
#             "size": top_k,
#             "query": {
#                 "script_score": {
#                     "query": {"match_all": {}},
#                     "script": {
#                         "source": "Math.max(cosineSimilarity(params.query_vector, 'embedding') + 1.0, 0.001)",
#                         "params": {"query_vector": query_embedding.tolist()}
#                     }
#                 }
#             }
#         }

#         # Add textbook filter if specified
#         if textbook_filter:
#             es_query["query"]["script_score"]["query"] = {
#                 "term": {"metadata.textbook_id": textbook_filter}
#             }

#         # Execute search
#         response = self.es.search(index=self.index_name, body=es_query)

#         # Process results
#         results = []
#         for hit in response['hits']['hits']:
#             source = hit['_source']

#             # Reconstruct Document object
#             doc = Document(
#                 page_content=source['content'],
#                 metadata=source['metadata']
#             )

#             result = {
#                 'document': doc,
#                 'score': hit['_score'],
#                 'content': source['content'],
#                 'metadata': source['metadata'],
#                 'citation': self._format_citation(doc)
#             }
#             results.append(result)

#         return results

#     def hybrid_search(self, query: str, top_k: int = 5,
#                       vector_weight: float = 0.7, text_weight: float = 0.3,
#                       textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
#         """Hybrid search combining vector similarity and text matching"""

#         # Generate query embedding
#         query_embedding = self.model.encode([query])[0]

#         # FIXED: Ensure non-negative scores in hybrid search too
#         es_query = {
#             "size": top_k,
#             "query": {
#                 "bool": {
#                     "should": [
#                         {
#                             "script_score": {
#                                 "query": {"match_all": {}},
#                                 "script": {
#                                     "source": f"Math.max(cosineSimilarity(params.query_vector, 'embedding'), 0.0) * {vector_weight}",
#                                     "params": {"query_vector": query_embedding.tolist()}
#                                 }
#                             }
#                         },
#                         {
#                             "multi_match": {
#                                 "query": query,
#                                 "fields": ["content^2", "metadata.chapter"],
#                                 "boost": text_weight
#                             }
#                         }
#                     ]
#                 }
#             }
#         }

#         # Add textbook filter if specified
#         if textbook_filter:
#             es_query["query"]["bool"]["filter"] = [
#                 {"term": {"metadata.textbook_id": textbook_filter}}
#             ]

#         # Execute search
#         response = self.es.search(index=self.index_name, body=es_query)

#         # Process results (same as regular search)
#         results = []
#         for hit in response['hits']['hits']:
#             source = hit['_source']

#             doc = Document(
#                 page_content=source['content'],
#                 metadata=source['metadata']
#             )

#             result = {
#                 'document': doc,
#                 'score': hit['_score'],
#                 'content': source['content'],
#                 'metadata': source['metadata'],
#                 'citation': self._format_citation(doc)
#             }
#             results.append(result)

#         return results

#     def _format_citation(self, doc: Document) -> str:
#         """Format citation for a document"""
#         metadata = doc.metadata
#         textbook = metadata.get('textbook', 'Unknown Textbook')
#         chapter = metadata.get('chapter', 'Unknown Chapter')
#         page = metadata.get('page', 'Unknown')

#         return f"{textbook}, {chapter}, page {page}"

#     def get_textbook_list(self) -> List[str]:
#         """Get list of available textbooks"""
#         query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook_id",
#                         "size": 1000
#                     }
#                 }
#             }
#         }

#         response = self.es.search(index=self.index_name, body=query)
#         textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
#         return sorted(textbooks)

#     def get_statistics(self) -> Dict[str, Any]:
#         """Get search engine statistics"""
#         # Get total document count
#         count_response = self.es.count(index=self.index_name)
#         total_documents = count_response['count']

#         # Get textbook breakdown
#         aggs_query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook",
#                         "size": 1000
#                     }
#                 }
#             }
#         }

#         response = self.es.search(index=self.index_name, body=aggs_query)
#         textbook_counts = {
#             bucket['key']: bucket['doc_count']
#             for bucket in response['aggregations']['textbooks']['buckets']
#         }

#         return {
#             'total_documents': total_documents,
#             'total_textbooks': len(textbook_counts),
#             'textbook_breakdown': textbook_counts,
#             'index_built': True,
#             'embedding_dimension': self.embedding_dim,
#             'search_engine': 'Elasticsearch'
#         }

#     def delete_textbook(self, textbook_id: str) -> bool:
#         """Delete all documents for a specific textbook"""
#         query = {
#             "query": {
#                 "term": {
#                     "metadata.textbook_id": textbook_id
#                 }
#             }
#         }

#         response = self.es.delete_by_query(index=self.index_name, body=query)
#         deleted_count = response['deleted']

#         print(f"Deleted {deleted_count} documents for textbook: {textbook_id}")
#         return deleted_count > 0

"""
Elasticsearch Search with Embedding Backup
==========================================
Vector similarity search with disk backup for embeddings
"""
import json
import time
import pickle
import os
import numpy as np
from typing import List, Dict, Any, Optional
from langchain.schema import Document

try:
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk, BulkIndexError
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

from .models import TextbookCollection

class ElasticsearchSearchEngine:
    """Elasticsearch with vector similarity search and embedding backup"""
    
    def __init__(self,
                 model_name: str = "abhinand/MedEmbed-large-v0.1",
                 es_host: str = "localhost:9200",
                 index_name: str = "textbook_search",
                 create_index: bool = False):
        
        if not HAS_ELASTICSEARCH:
            raise ImportError("elasticsearch not installed. Run: pip install elasticsearch")
        
        if not HAS_SENTENCE_TRANSFORMERS:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.model_name = model_name
        self.index_name = index_name
        
        # Initialize Elasticsearch client with better settings
        self.es = Elasticsearch([es_host], request_timeout=60, max_retries=3)
        if not self.es.ping():
            raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")
        print(f"Connected to Elasticsearch at {es_host}")
        
        # Initialize sentence transformer
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Get actual embedding dimension from the model
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Model loaded, embedding dimension: {self.embedding_dim}")
        
        # Check if index exists and has correct dimension
        if self.es.indices.exists(index=self.index_name):
            mapping = self.es.indices.get_mapping(index=self.index_name)
            current_dims = mapping[self.index_name]['mappings']['properties']['embedding']['dims']
            
            if current_dims != self.embedding_dim:
                print(f"⚠️  WARNING: Index has {current_dims} dims but model has {self.embedding_dim} dims")
                print(f"⚠️  You MUST rebuild the index with force_rebuild=True")
                if create_index:
                    self._create_index_mapping()
            else:
                print(f"Using existing index: {self.index_name}")
                count = self.es.count(index=self.index_name)['count']
                print(f"Index contains {count} documents")
        else:
            self._create_index_mapping()
    
    def _create_index_mapping(self):
        """Create Elasticsearch index with proper mapping"""
        mapping = {
            "mappings": {
                "properties": {
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": self.embedding_dim,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {
                        "properties": {
                            "source": {"type": "keyword"},
                            "textbook_id": {"type": "keyword"},
                            "textbook": {"type": "keyword"},
                            "chapter": {"type": "text"},
                            "chapter_start_page": {"type": "integer"},
                            "chapter_end_page": {"type": "integer"},
                            "page": {"type": "integer"},
                            "chunk_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"}
                        }
                    }
                }
            },
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "index.max_result_window": 10000
            }
        }
        
        if self.es.indices.exists(index=self.index_name):
            print(f"Deleting existing index: {self.index_name}")
            self.es.indices.delete(index=self.index_name)
        
        print(f"Creating index: {self.index_name} with {self.embedding_dim} dimensions")
        self.es.indices.create(index=self.index_name, body=mapping)
    
    def build_index_from_collection(self, collection: TextbookCollection, force_rebuild: bool = False):
        """Build search index from textbook collection"""
        if force_rebuild:
            self._create_index_mapping()
        return self._index_documents(collection.all_documents)
    
    def build_index_from_documents(self, documents: List[Document],
                                    cache_name: str = None, force_rebuild: bool = False):
        """Build search index from document list"""
        if force_rebuild:
            self._create_index_mapping()
        return self._index_documents(documents)
    
    def _save_embeddings_backup(self, embeddings, backup_path: str):
        """Save embeddings to disk for backup"""
        print(f"\n💾 Saving embeddings backup to {backup_path}...")
        
        backup_data = {
            'embeddings': embeddings,
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'total_count': len(embeddings),
            'timestamp': time.time()
        }
        
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        with open(backup_path, 'wb') as f:
            pickle.dump(backup_data, f)
        
        file_size_mb = os.path.getsize(backup_path) / (1024*1024)
        print(f"✅ Saved {len(embeddings)} embeddings to disk")
        print(f"   File: {backup_path}")
        print(f"   Size: {file_size_mb:.2f} MB")
    
    def _load_embeddings_backup(self, backup_path: str, expected_count: int) -> Optional[np.ndarray]:
        """Load embeddings from disk backup"""
        if not os.path.exists(backup_path):
            print("📂 No embedding backup found, will generate fresh embeddings")
            return None
        
        print(f"\n📂 Found embeddings backup: {backup_path}")
        
        try:
            with open(backup_path, 'rb') as f:
                backup_data = pickle.load(f)
            
            # Validate backup
            if backup_data['total_count'] != expected_count:
                print(f"⚠️  Backup has {backup_data['total_count']} embeddings but need {expected_count}")
                print("   Will regenerate embeddings")
                return None
            
            if backup_data['model_name'] != self.model_name:
                print(f"⚠️  Backup uses model '{backup_data['model_name']}' but current model is '{self.model_name}'")
                print("   Will regenerate embeddings")
                return None
            
            if backup_data['embedding_dim'] != self.embedding_dim:
                print(f"⚠️  Backup has dimension {backup_data['embedding_dim']} but need {self.embedding_dim}")
                print("   Will regenerate embeddings")
                return None
            
            file_size_mb = os.path.getsize(backup_path) / (1024*1024)
            print(f"✅ Loaded {backup_data['total_count']} embeddings from backup ({file_size_mb:.2f} MB)")
            print(f"   Model: {backup_data['model_name']}")
            print(f"   Dimension: {backup_data['embedding_dim']}")
            
            return backup_data['embeddings']
            
        except Exception as e:
            print(f"⚠️  Error loading backup: {e}")
            print("   Will regenerate embeddings")
            return None
    
    def _index_documents(self, documents: List[Document]) -> bool:
        """Index documents in Elasticsearch with backup and error handling"""
        print(f"Indexing {len(documents)} documents...")
        
        # Try to load existing embeddings from disk backup
        backup_path = "data/embeddings_backup.pkl"
        embeddings = self._load_embeddings_backup(backup_path, len(documents))
        
        if embeddings is None:
            # Generate embeddings
            print("Generating embeddings...")
            start_time = time.time()
            texts = [doc.page_content for doc in documents]
            embeddings = self.model.encode(texts, show_progress_bar=True)
            embed_time = time.time() - start_time
            print(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
            # Save embeddings backup to disk
            self._save_embeddings_backup(embeddings, backup_path)
        else:
            print("✅ Using embeddings from backup (saved generation time!)")
        
        # Prepare documents for bulk indexing with validation
        print("\nPreparing documents for indexing...")
        actions = []
        skipped = 0
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            try:
                # Validate and clean content
                content = doc.page_content
                if not content or len(content.strip()) == 0:
                    skipped += 1
                    continue
                
                # Truncate very long content (Elasticsearch limit)
                if len(content) > 50000:
                    content = content[:50000]
                
                # Validate embedding
                if embedding is None or len(embedding) != self.embedding_dim:
                    print(f"⚠️  Skipping doc {i}: invalid embedding")
                    skipped += 1
                    continue
                
                action = {
                    "_index": self.index_name,
                    "_id": f"doc_{i}",
                    "_source": {
                        "content": content,
                        "embedding": embedding.tolist(),
                        "metadata": doc.metadata
                    }
                }
                actions.append(action)
                
            except Exception as e:
                print(f"⚠️  Error preparing doc {i}: {e}")
                skipped += 1
                continue
        
        if skipped > 0:
            print(f"⚠️  Skipped {skipped} invalid documents")
        
        print(f"\nBulk indexing {len(actions)} documents...")
        
        # Bulk index with smaller batches and error handling
        indexed_count = 0
        failed_count = 0
        batch_size = 500  # Smaller batches to avoid timeouts
        
        for batch_start in range(0, len(actions), batch_size):
            batch_end = min(batch_start + batch_size, len(actions))
            batch = actions[batch_start:batch_end]
            
            try:
                success, failed = bulk(
                    self.es,
                    batch,
                    chunk_size=100,
                    request_timeout=60,
                    raise_on_error=False,
                    raise_on_exception=False
                )
                
                indexed_count += success
                
                if failed:
                    failed_count += len(failed)
                    # Log first few failures for debugging
                    for error in failed[:3]:
                        print(f"⚠️  Failed to index: {error}")
                
                # Progress update every 10 batches
                if (batch_start // batch_size) % 10 == 0:
                    progress = (batch_end / len(actions)) * 100
                    print(f"Progress: {progress:.1f}% ({batch_end}/{len(actions)})")
                    
            except BulkIndexError as e:
                print(f"⚠️  Bulk indexing error in batch {batch_start}-{batch_end}")
                print(f"   {len(e.errors)} documents failed")
                failed_count += len(batch)
                
                # Log first few errors for debugging
                for error in e.errors[:3]:
                    print(f"   Error: {error}")
                    
            except Exception as e:
                print(f"❌ Unexpected error in batch {batch_start}-{batch_end}: {e}")
                failed_count += len(batch)
        
        # Refresh index
        self.es.indices.refresh(index=self.index_name)
        
        print(f"\n✅ Indexing complete!")
        print(f"   Successfully indexed: {indexed_count}")
        print(f"   Failed: {failed_count}")
        print(f"   Skipped: {skipped}")
        print(f"   Total in index: {indexed_count}")
        
        return indexed_count > 0
    
    def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Vector similarity search
        
        Args:
            query: The search query
            top_k: Number of results to return
            textbook_filter: Optional textbook ID to filter by
        """
        query_embedding = self.model.encode([query])[0]
        
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
        
        if textbook_filter:
            es_query["query"]["script_score"]["query"] = {
                "term": {"metadata.textbook_id": textbook_filter}
            }
        
        response = self.es.search(index=self.index_name, body=es_query)
        
        results = []
        for hit in response['hits']['hits']:
            source = hit['_source']
            doc = Document(page_content=source['content'], metadata=source['metadata'])
            result = {
                'document': doc,
                'score': hit['_score'],
                'content': source['content'],
                'metadata': source['metadata'],
                'citation': self._format_citation(doc)
            }
            results.append(result)
        
        return results
    
    def _format_citation(self, doc: Document) -> str:
        """Format citation for a document"""
        metadata = doc.metadata
        textbook = metadata.get('textbook', 'Unknown Textbook')
        chapter = metadata.get('chapter', 'Unknown Chapter')
        page = metadata.get('page', 'Unknown')
        return f"{textbook}, {chapter}, page {page}"
    
    def get_textbook_list(self) -> List[str]:
        """Get list of available textbooks"""
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
        
        response = self.es.search(index=self.index_name, body=query)
        textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
        return sorted(textbooks)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics"""
        count_response = self.es.count(index=self.index_name)
        total_documents = count_response['count']
        
        aggs_query = {
            "size": 0,
            "aggs": {
                "textbooks": {
                    "terms": {
                        "field": "metadata.textbook",
                        "size": 1000
                    }
                }
            }
        }
        
        response = self.es.search(index=self.index_name, body=aggs_query)
        textbook_counts = {
            bucket['key']: bucket['doc_count']
            for bucket in response['aggregations']['textbooks']['buckets']
        }
        
        return {
            'total_documents': total_documents,
            'total_textbooks': len(textbook_counts),
            'textbook_breakdown': textbook_counts,
            'index_built': True,
            'embedding_dimension': self.embedding_dim,
            'model_name': self.model_name,
            'search_engine': 'Elasticsearch'
        }
    
    def delete_textbook(self, textbook_id: str) -> bool:
        """Delete all documents for a specific textbook"""
        query = {
            "query": {
                "term": {
                    "metadata.textbook_id": textbook_id
                }
            }
        }
        
        response = self.es.delete_by_query(index=self.index_name, body=query)
        deleted_count = response['deleted']
        print(f"Deleted {deleted_count} documents for textbook: {textbook_id}")
        return deleted_count > 0



# """
# Elasticsearch Search with Hybrid Search and Query Expansion
# ===========================================================
# Vector similarity + BM25 keyword search with medical query expansion
# """
# import json
# import time
# import pickle
# import os
# import re
# import numpy as np
# from typing import List, Dict, Any, Optional
# from langchain.schema import Document

# try:
#     from elasticsearch import Elasticsearch
#     from elasticsearch.helpers import bulk, BulkIndexError
#     HAS_ELASTICSEARCH = True
# except ImportError:
#     HAS_ELASTICSEARCH = False

# try:
#     from sentence_transformers import SentenceTransformer
#     HAS_SENTENCE_TRANSFORMERS = True
# except ImportError:
#     HAS_SENTENCE_TRANSFORMERS = False

# from .models import TextbookCollection

# class ElasticsearchSearchEngine:
#     """Elasticsearch with hybrid search (semantic + keyword) and query expansion"""
    
#     def __init__(self,
#                  model_name: str = "abhinand/MedEmbed-large-v0.1",
#                  es_host: str = "localhost:9200",
#                  index_name: str = "textbook_search",
#                  create_index: bool = False):
        
#         if not HAS_ELASTICSEARCH:
#             raise ImportError("elasticsearch not installed. Run: pip install elasticsearch")
        
#         if not HAS_SENTENCE_TRANSFORMERS:
#             raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
#         self.model_name = model_name
#         self.index_name = index_name
        
#         # Initialize Elasticsearch client with better settings
#         self.es = Elasticsearch([es_host], request_timeout=60, max_retries=3)
#         if not self.es.ping():
#             raise ConnectionError(f"Could not connect to Elasticsearch at {es_host}")
#         print(f"Connected to Elasticsearch at {es_host}")
        
#         # Initialize sentence transformer
#         print(f"Loading model: {model_name}")
#         self.model = SentenceTransformer(model_name)
        
#         # Get actual embedding dimension from the model
#         self.embedding_dim = self.model.get_sentence_embedding_dimension()
#         print(f"Model loaded, embedding dimension: {self.embedding_dim}")
        
#         # Check if index exists and has correct dimension
#         if self.es.indices.exists(index=self.index_name):
#             mapping = self.es.indices.get_mapping(index=self.index_name)
#             current_dims = mapping[self.index_name]['mappings']['properties']['embedding']['dims']
            
#             if current_dims != self.embedding_dim:
#                 print(f"⚠️  WARNING: Index has {current_dims} dims but model has {self.embedding_dim} dims")
#                 print(f"⚠️  You MUST rebuild the index with force_rebuild=True")
#                 if create_index:
#                     self._create_index_mapping()
#             else:
#                 print(f"Using existing index: {self.index_name}")
#                 count = self.es.count(index=self.index_name)['count']
#                 print(f"Index contains {count} documents")
#         else:
#             self._create_index_mapping()
    
#     def _create_index_mapping(self):
#         """Create Elasticsearch index with proper mapping"""
#         mapping = {
#             "mappings": {
#                 "properties": {
#                     "content": {
#                         "type": "text",
#                         "analyzer": "standard"
#                     },
#                     "embedding": {
#                         "type": "dense_vector",
#                         "dims": self.embedding_dim,
#                         "index": True,
#                         "similarity": "cosine"
#                     },
#                     "metadata": {
#                         "properties": {
#                             "source": {"type": "keyword"},
#                             "textbook_id": {"type": "keyword"},
#                             "textbook": {"type": "keyword"},
#                             "chapter": {"type": "text"},
#                             "chapter_start_page": {"type": "integer"},
#                             "chapter_end_page": {"type": "integer"},
#                             "page": {"type": "integer"},
#                             "chunk_id": {"type": "keyword"},
#                             "chunk_index": {"type": "integer"}
#                         }
#                     }
#                 }
#             },
#             "settings": {
#                 "number_of_shards": 1,
#                 "number_of_replicas": 0,
#                 "index.max_result_window": 10000
#             }
#         }
        
#         if self.es.indices.exists(index=self.index_name):
#             print(f"Deleting existing index: {self.index_name}")
#             self.es.indices.delete(index=self.index_name)
        
#         print(f"Creating index: {self.index_name} with {self.embedding_dim} dimensions")
#         self.es.indices.create(index=self.index_name, body=mapping)
    
#     def build_index_from_collection(self, collection: TextbookCollection, force_rebuild: bool = False):
#         """Build search index from textbook collection"""
#         if force_rebuild:
#             self._create_index_mapping()
#         return self._index_documents(collection.all_documents)
    
#     def build_index_from_documents(self, documents: List[Document],
#                                     cache_name: str = None, force_rebuild: bool = False):
#         """Build search index from document list"""
#         if force_rebuild:
#             self._create_index_mapping()
#         return self._index_documents(documents)
    
#     def _save_embeddings_backup(self, embeddings, backup_path: str):
#         """Save embeddings to disk for backup"""
#         print(f"\n💾 Saving embeddings backup to {backup_path}...")
        
#         backup_data = {
#             'embeddings': embeddings,
#             'model_name': self.model_name,
#             'embedding_dim': self.embedding_dim,
#             'total_count': len(embeddings),
#             'timestamp': time.time()
#         }
        
#         os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
#         with open(backup_path, 'wb') as f:
#             pickle.dump(backup_data, f)
        
#         file_size_mb = os.path.getsize(backup_path) / (1024*1024)
#         print(f"✅ Saved {len(embeddings)} embeddings to disk")
#         print(f"   File: {backup_path}")
#         print(f"   Size: {file_size_mb:.2f} MB")
    
#     def _load_embeddings_backup(self, backup_path: str, expected_count: int) -> Optional[np.ndarray]:
#         """Load embeddings from disk backup"""
#         if not os.path.exists(backup_path):
#             print("📂 No embedding backup found, will generate fresh embeddings")
#             return None
        
#         print(f"\n📂 Found embeddings backup: {backup_path}")
        
#         try:
#             with open(backup_path, 'rb') as f:
#                 backup_data = pickle.load(f)
            
#             # Validate backup
#             if backup_data['total_count'] != expected_count:
#                 print(f"⚠️  Backup has {backup_data['total_count']} embeddings but need {expected_count}")
#                 print("   Will regenerate embeddings")
#                 return None
            
#             if backup_data['model_name'] != self.model_name:
#                 print(f"⚠️  Backup uses model '{backup_data['model_name']}' but current model is '{self.model_name}'")
#                 print("   Will regenerate embeddings")
#                 return None
            
#             if backup_data['embedding_dim'] != self.embedding_dim:
#                 print(f"⚠️  Backup has dimension {backup_data['embedding_dim']} but need {self.embedding_dim}")
#                 print("   Will regenerate embeddings")
#                 return None
            
#             file_size_mb = os.path.getsize(backup_path) / (1024*1024)
#             print(f"✅ Loaded {backup_data['total_count']} embeddings from backup ({file_size_mb:.2f} MB)")
#             print(f"   Model: {backup_data['model_name']}")
#             print(f"   Dimension: {backup_data['embedding_dim']}")
            
#             return backup_data['embeddings']
            
#         except Exception as e:
#             print(f"⚠️  Error loading backup: {e}")
#             print("   Will regenerate embeddings")
#             return None
    
#     def _index_documents(self, documents: List[Document]) -> bool:
#         """Index documents in Elasticsearch with backup and error handling"""
#         print(f"Indexing {len(documents)} documents...")
        
#         # Try to load existing embeddings from disk backup
#         backup_path = "data/embeddings_backup.pkl"
#         embeddings = self._load_embeddings_backup(backup_path, len(documents))
        
#         if embeddings is None:
#             # Generate embeddings
#             print("Generating embeddings...")
#             start_time = time.time()
#             texts = [doc.page_content for doc in documents]
#             embeddings = self.model.encode(texts, show_progress_bar=True)
#             embed_time = time.time() - start_time
#             print(f"Generated {len(embeddings)} embeddings in {embed_time:.2f}s")
            
#             # Save embeddings backup to disk
#             self._save_embeddings_backup(embeddings, backup_path)
#         else:
#             print("✅ Using embeddings from backup (saved generation time!)")
        
#         # Prepare documents for bulk indexing with validation
#         print("\nPreparing documents for indexing...")
#         actions = []
#         skipped = 0
        
#         for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#             try:
#                 # Validate and clean content
#                 content = doc.page_content
#                 if not content or len(content.strip()) == 0:
#                     skipped += 1
#                     continue
                
#                 # Truncate very long content (Elasticsearch limit)
#                 if len(content) > 50000:
#                     content = content[:50000]
                
#                 # Validate embedding
#                 if embedding is None or len(embedding) != self.embedding_dim:
#                     print(f"⚠️  Skipping doc {i}: invalid embedding")
#                     skipped += 1
#                     continue
                
#                 action = {
#                     "_index": self.index_name,
#                     "_id": f"doc_{i}",
#                     "_source": {
#                         "content": content,
#                         "embedding": embedding.tolist(),
#                         "metadata": doc.metadata
#                     }
#                 }
#                 actions.append(action)
                
#             except Exception as e:
#                 print(f"⚠️  Error preparing doc {i}: {e}")
#                 skipped += 1
#                 continue
        
#         if skipped > 0:
#             print(f"⚠️  Skipped {skipped} invalid documents")
        
#         print(f"\nBulk indexing {len(actions)} documents...")
        
#         # Bulk index with smaller batches and error handling
#         indexed_count = 0
#         failed_count = 0
#         batch_size = 500  # Smaller batches to avoid timeouts
        
#         for batch_start in range(0, len(actions), batch_size):
#             batch_end = min(batch_start + batch_size, len(actions))
#             batch = actions[batch_start:batch_end]
            
#             try:
#                 success, failed = bulk(
#                     self.es,
#                     batch,
#                     chunk_size=100,
#                     request_timeout=60,
#                     raise_on_error=False,
#                     raise_on_exception=False
#                 )
                
#                 indexed_count += success
                
#                 if failed:
#                     failed_count += len(failed)
#                     # Log first few failures for debugging
#                     for error in failed[:3]:
#                         print(f"⚠️  Failed to index: {error}")
                
#                 # Progress update every 10 batches
#                 if (batch_start // batch_size) % 10 == 0:
#                     progress = (batch_end / len(actions)) * 100
#                     print(f"Progress: {progress:.1f}% ({batch_end}/{len(actions)})")
                    
#             except BulkIndexError as e:
#                 print(f"⚠️  Bulk indexing error in batch {batch_start}-{batch_end}")
#                 print(f"   {len(e.errors)} documents failed")
#                 failed_count += len(batch)
                
#                 # Log first few errors for debugging
#                 for error in e.errors[:3]:
#                     print(f"   Error: {error}")
                    
#             except Exception as e:
#                 print(f"❌ Unexpected error in batch {batch_start}-{batch_end}: {e}")
#                 failed_count += len(batch)
        
#         # Refresh index
#         self.es.indices.refresh(index=self.index_name)
        
#         print(f"\n✅ Indexing complete!")
#         print(f"   Successfully indexed: {indexed_count}")
#         print(f"   Failed: {failed_count}")
#         print(f"   Skipped: {skipped}")
#         print(f"   Total in index: {indexed_count}")
        
#         return indexed_count > 0
    
#     def expand_query_medical(self, question: str) -> List[str]:
#         """
#         Generate medical query variations for better retrieval
        
#         Strategies:
#         1. Original full question
#         2. Simplified version (remove multiple choice preamble)
#         3. Extract key clinical scenario
#         4. Focus on diagnosis/condition keywords
#         """
#         queries = [question]
        
#         # Strategy 1: Remove "Which of the following" preamble
#         simplified = re.sub(r'Which of the following.*?\?', '', question, flags=re.IGNORECASE)
#         simplified = re.sub(r'What is the (MOST|most).*?\?', '', simplified, flags=re.IGNORECASE)
#         simplified = simplified.strip()
        
#         if len(simplified) > 50:
#             queries.append(simplified)
        
#         # Strategy 2: Extract clinical scenario (before the question)
#         question_markers = [
#             'Which of the following',
#             'What is the MOST',
#             'What is the most',
#             'Which medication',
#             'Which drug',
#             'What medication'
#         ]
        
#         for marker in question_markers:
#             if marker in question:
#                 before_question = question.split(marker)[0].strip()
#                 if len(before_question) > 100:
#                     queries.append(before_question)
#                 break
        
#         # Strategy 3: Extract diagnosis keywords
#         diagnosis_patterns = [
#             r'diagnosis\?',
#             r'condition\?',
#             r'complication\?',
#             r'cause\?',
#             r'mechanism\?',
#             r'risk factor\?',
#             r'treatment\?',
#             r'medication\?'
#         ]
        
#         for pattern in diagnosis_patterns:
#             match = re.search(pattern, question, re.IGNORECASE)
#             if match:
#                 before_pattern = question[:match.start()].strip()
#                 # Extract last sentence before the question
#                 last_sentence = before_pattern.split('.')[-1].strip()
#                 if len(last_sentence) > 30:
#                     queries.append(last_sentence)
#                 break
        
#         # Strategy 4: Extract key medical terms (symptoms, conditions)
#         # Focus on phrases after "presents with", "history of", etc.
#         medical_context_markers = [
#             'presents with',
#             'history of',
#             'examination reveals',
#             'develops',
#             'diagnosed with',
#             'shows',
#             'reports'
#         ]
        
#         for marker in medical_context_markers:
#             if marker in question.lower():
#                 parts = re.split(marker, question, flags=re.IGNORECASE)
#                 if len(parts) > 1:
#                     # Get the part after the marker, take first sentence
#                     after_marker = parts[1].split('.')[0].strip()
#                     if len(after_marker) > 20:
#                         queries.append(f"{marker} {after_marker}")
        
#         # Remove duplicates while preserving order
#         seen = set()
#         unique_queries = []
#         for q in queries:
#             q_clean = q.lower().strip()
#             if q_clean and q_clean not in seen and len(q_clean) > 20:
#                 seen.add(q_clean)
#                 unique_queries.append(q)
        
#         # Limit to top 3 most promising queries
#         return unique_queries[:3]
    
#     def _reciprocal_rank_fusion(self, all_results: List[Dict], k: int = 60) -> List[Dict]:
#         """
#         Combine results from multiple queries using Reciprocal Rank Fusion
        
#         Args:
#             all_results: List of (query_idx, rank, result) tuples
#             k: RRF constant (default 60)
        
#         Returns:
#             Sorted list of results by RRF score
#         """
#         rrf_scores = {}
#         result_map = {}
        
#         for query_idx, rank, result in all_results:
#             chunk_id = result['metadata'].get('chunk_id', result['metadata'].get('source', ''))
            
#             # RRF score: 1 / (k + rank)
#             score = 1.0 / (k + rank)
            
#             if chunk_id not in rrf_scores:
#                 rrf_scores[chunk_id] = 0
#                 result_map[chunk_id] = result
            
#             rrf_scores[chunk_id] += score
        
#         # Sort by RRF score
#         sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
#         # Build final results with RRF scores
#         final_results = []
#         for chunk_id, rrf_score in sorted_items:
#             result = result_map[chunk_id].copy()
#             result['score'] = rrf_score  # Replace with RRF score
#             final_results.append(result)
        
#         return final_results
    
#     def search(self, query: str, top_k: int = 5, textbook_filter: Optional[str] = None,
#                use_hybrid: bool = True, use_query_expansion: bool = True,
#                semantic_weight: float = 0.5) -> List[Dict[str, Any]]:
#         """
#         Enhanced search with hybrid retrieval and query expansion
        
#         Args:
#             query: The search query
#             top_k: Number of results to return
#             textbook_filter: Optional textbook ID to filter by
#             use_hybrid: Use hybrid search (semantic + BM25)
#             use_query_expansion: Expand query for better retrieval
#             semantic_weight: Weight for semantic vs keyword (0-1, only used if hybrid=False)
        
#         Returns:
#             List of search results with scores and metadata
#         """
        
#         # Query expansion
#         if use_query_expansion:
#             query_variants = self.expand_query_medical(query)
#             print(f"🔍 Query expansion: {len(query_variants)} variants")
#             for i, variant in enumerate(query_variants, 1):
#                 print(f"   [{i}] {variant[:100]}...")
#         else:
#             query_variants = [query]
        
#         # Retrieve results for each query variant
#         all_results = []
        
#         for query_idx, q in enumerate(query_variants):
#             if use_hybrid:
#                 # HYBRID SEARCH: Semantic + BM25
#                 query_embedding = self.model.encode([q])[0]
                
#                 es_query = {
#                     "size": top_k * 3,  # Get more for fusion
#                     "query": {
#                         "bool": {
#                             "should": [
#                                 # Semantic vector search
#                                 {
#                                     "script_score": {
#                                         "query": {"match_all": {}},
#                                         "script": {
#                                             "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
#                                             "params": {"query_vector": query_embedding.tolist()}
#                                         },
#                                         "boost": 1.0
#                                     }
#                                 },
#                                 # BM25 keyword search on content
#                                 {
#                                     "match": {
#                                         "content": {
#                                             "query": q,
#                                             "boost": 1.2
#                                         }
#                                     }
#                                 },
#                                 # BM25 on chapter (lower weight)
#                                 {
#                                     "match": {
#                                         "metadata.chapter": {
#                                             "query": q,
#                                             "boost": 0.5
#                                         }
#                                     }
#                                 }
#                             ],
#                             "minimum_should_match": 1
#                         }
#                     }
#                 }
                
#                 if textbook_filter:
#                     es_query["query"]["bool"]["filter"] = [
#                         {"term": {"metadata.textbook_id": textbook_filter}}
#                     ]
                
#             else:
#                 # Pure vector search (original)
#                 query_embedding = self.model.encode([q])[0]
                
#                 es_query = {
#                     "size": top_k * 3,
#                     "query": {
#                         "script_score": {
#                             "query": {"match_all": {}},
#                             "script": {
#                                 "source": "Math.max(cosineSimilarity(params.query_vector, 'embedding') + 1.0, 0.001)",
#                                 "params": {"query_vector": query_embedding.tolist()}
#                             }
#                         }
#                     }
#                 }
                
#                 if textbook_filter:
#                     es_query["query"]["script_score"]["query"] = {
#                         "term": {"metadata.textbook_id": textbook_filter}
#                     }
            
#             response = self.es.search(index=self.index_name, body=es_query)
            
#             # Collect results with rank
#             for rank, hit in enumerate(response['hits']['hits'], 1):
#                 source = hit['_source']
#                 doc = Document(page_content=source['content'], metadata=source['metadata'])
#                 result = {
#                     'document': doc,
#                     'score': hit['_score'],
#                     'content': source['content'],
#                     'metadata': source['metadata'],
#                     'citation': self._format_citation(doc)
#                 }
#                 all_results.append((query_idx, rank, result))
        
#         # Apply Reciprocal Rank Fusion if using query expansion
#         if use_query_expansion and len(query_variants) > 1:
#             print(f"📊 Applying Reciprocal Rank Fusion across {len(query_variants)} queries")
#             final_results = self._reciprocal_rank_fusion(all_results)
#         else:
#             # Just sort by score
#             final_results = sorted([r[2] for r in all_results], 
#                                   key=lambda x: x['score'], reverse=True)
        
#         # Return top K unique results
#         seen_chunks = set()
#         unique_results = []
        
#         for result in final_results:
#             chunk_id = result['metadata'].get('chunk_id', result['metadata'].get('source', ''))
#             if chunk_id not in seen_chunks:
#                 unique_results.append(result)
#                 seen_chunks.add(chunk_id)
            
#             if len(unique_results) >= top_k:
#                 break
        
#         return unique_results
    
#     def _format_citation(self, doc: Document) -> str:
#         """Format citation for a document"""
#         metadata = doc.metadata
#         textbook = metadata.get('textbook', 'Unknown Textbook')
#         chapter = metadata.get('chapter', 'Unknown Chapter')
#         page = metadata.get('page', 'Unknown')
#         return f"{textbook}, {chapter}, page {page}"
    
#     def get_textbook_list(self) -> List[str]:
#         """Get list of available textbooks"""
#         query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook_id",
#                         "size": 1000
#                     }
#                 }
#             }
#         }
        
#         response = self.es.search(index=self.index_name, body=query)
#         textbooks = [bucket['key'] for bucket in response['aggregations']['textbooks']['buckets']]
#         return sorted(textbooks)
    
#     def get_statistics(self) -> Dict[str, Any]:
#         """Get search engine statistics"""
#         count_response = self.es.count(index=self.index_name)
#         total_documents = count_response['count']
        
#         aggs_query = {
#             "size": 0,
#             "aggs": {
#                 "textbooks": {
#                     "terms": {
#                         "field": "metadata.textbook",
#                         "size": 1000
#                     }
#                 }
#             }
#         }
        
#         response = self.es.search(index=self.index_name, body=aggs_query)
#         textbook_counts = {
#             bucket['key']: bucket['doc_count']
#             for bucket in response['aggregations']['textbooks']['buckets']
#         }
        
#         return {
#             'total_documents': total_documents,
#             'total_textbooks': len(textbook_counts),
#             'textbook_breakdown': textbook_counts,
#             'index_built': True,
#             'embedding_dimension': self.embedding_dim,
#             'model_name': self.model_name,
#             'search_engine': 'Elasticsearch (Hybrid + Query Expansion)'
#         }
    
#     def delete_textbook(self, textbook_id: str) -> bool:
#         """Delete all documents for a specific textbook"""
#         query = {
#             "query": {
#                 "term": {
#                     "metadata.textbook_id": textbook_id
#                 }
#             }
#         }
        
#         response = self.es.delete_by_query(index=self.index_name, body=query)
#         deleted_count = response['deleted']
#         print(f"Deleted {deleted_count} documents for textbook: {textbook_id}")
#         return deleted_count > 0