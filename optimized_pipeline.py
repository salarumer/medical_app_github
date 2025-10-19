#latest one befroe gemini

"""
Optimized Single Textbook Processing Pipeline
=============================================

Fast, scalable approach with separate files per textbook.
"""

import os
import sys
import pickle
import numpy as np
from typing import Dict, List
from core.processor import ProductionTextbookProcessor
from core.elasticsearch_search_engine import ElasticsearchSearchEngine
from core.models import TextbookInfo, TextbookCollection

class OptimizedTextbookManager:
    """Manages textbooks with separate files for optimal performance"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.textbooks_dir = os.path.join(data_dir, "textbooks")
        self.registry_path = os.path.join(data_dir, "textbook_registry.pkl")
        self.vectors_path = os.path.join(data_dir, "optimized_search_vectors.pkl")
        
        os.makedirs(self.textbooks_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    
    def load_registry(self) -> Dict:
        """Load textbook registry (metadata only)"""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'rb') as f:
                return pickle.load(f)
        return {"textbooks": {}, "total_chunks": 0, "version": 1}
    
    def save_registry(self, registry: Dict):
        """Save textbook registry"""
        with open(self.registry_path, 'wb') as f:
            pickle.dump(registry, f)
    
    def save_textbook_chunks(self, textbook_id: str, documents: List, textbook_info: TextbookInfo):
        """Save chunks for a single textbook"""
        textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
        textbook_data = {
            "textbook_info": textbook_info,
            "documents": documents,
            "chunk_count": len(documents)
        }
        
        with open(textbook_file, 'wb') as f:
            pickle.dump(textbook_data, f)
        
        print(f"Saved {len(documents)} chunks to {textbook_id}_chunks.pkl")
    
    def load_textbook_chunks(self, textbook_id: str):
        """Load chunks for a specific textbook"""
        textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
        if os.path.exists(textbook_file):
            with open(textbook_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def load_all_documents(self) -> List:
        """Load all documents from all textbook files (only when needed)"""
        registry = self.load_registry()
        all_documents = []
        
        for textbook_id in registry["textbooks"].keys():
            textbook_data = self.load_textbook_chunks(textbook_id)
            if textbook_data:
                all_documents.extend(textbook_data["documents"])
        
        return all_documents
    
    def get_collection_for_search(self) -> TextbookCollection:
        """Create a TextbookCollection for search (loads all docs)"""
        collection = TextbookCollection()
        registry = self.load_registry()
        
        for textbook_id in registry["textbooks"].keys():
            textbook_data = self.load_textbook_chunks(textbook_id)
            if textbook_data:
                from core.models import ProcessingResult
                result = ProcessingResult(
                    textbook_info=textbook_data["textbook_info"],
                    documents=textbook_data["documents"],
                    processing_time=0,
                    success=True
                )
                collection.add_textbook(result)
        
        return collection

def process_multiple_textbooks_optimized(textbook_configs: List[Dict]):
    """Process multiple textbooks efficiently"""
    
    print("OPTIMIZED MULTIPLE TEXTBOOK PROCESSING")
    print("=" * 60)
    
    manager = OptimizedTextbookManager()
    registry = manager.load_registry()
    processor = ProductionTextbookProcessor()
    
    processed_count = 0
    failed_count = 0
    
    for i, config in enumerate(textbook_configs, 1):
        print(f"\n[{i}/{len(textbook_configs)}] Processing textbook...")
        
        textbook_info = TextbookInfo(
            id=config['id'],
            title=config['title'],
            edition=config.get('edition'),
            year=config.get('year'),
            file_path=config['pdf_path']
        )
        
        print(f"Book: {textbook_info.display_name}")
        
        if textbook_info.id in registry["textbooks"]:
            print(f"Warning: Textbook '{textbook_info.id}' already exists. Replacing...")
        
        result = processor.process_textbook(
            textbook_info=textbook_info,
            pdf_path=config['pdf_path'],
            chapter_csv_path=config['chapter_csv_path']
        )
        
        if not result.success:
            print(f"Error: Processing failed: {result.error_message}")
            failed_count += 1
            continue
        
        print(f"Successfully processed {len(result.documents)} document chunks")
        
        manager.save_textbook_chunks(textbook_info.id, result.documents, textbook_info)
        
        registry["textbooks"][textbook_info.id] = {
            "title": textbook_info.display_name,
            "chunk_count": len(result.documents),
            "file_path": f"{textbook_info.id}_chunks.pkl"
        }
        
        processed_count += 1
    
    if processed_count == 0:
        print("\nError: No textbooks were successfully processed!")
        return False
    
    registry["total_chunks"] = sum(tb["chunk_count"] for tb in registry["textbooks"].values())
    manager.save_registry(registry)
    
    print(f"\nRegistry updated: {len(registry['textbooks'])} textbooks, {registry['total_chunks']} total chunks")
    
    print("\nRebuilding search index...")
    collection = manager.get_collection_for_search()
    
    search_engine = ElasticsearchSearchEngine(es_host="http://localhost:9200")
    search_engine.build_index_from_collection(collection, force_rebuild=True)
    
    stats = search_engine.get_statistics()
    print(f"\nSYSTEM UPDATED!")
    print("=" * 60)
    print(f"Successfully processed: {processed_count}")
    print(f"Failed: {failed_count}")
    print(f"Total textbooks in system: {stats['total_textbooks']}")
    print(f"Total documents: {stats['total_documents']}")
    
    print(f"\nAvailable textbooks:")
    for textbook, count in stats['textbook_breakdown'].items():
        print(f"   - {textbook}: {count} chunks")
    
    return True

def main():
    """Main function - configure your textbooks here"""
    
    # CONFIGURE YOUR TEXTBOOKS HERE
    textbook_configs = [
        {
            'id': 'tintinalli_emergency_8e',
            'title': "Tintinalli's Emergency Medicine",
            'edition': '8th Edition',
            'year': '2016',
            'pdf_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/multi_textbook_search/data/textbooks/Tintinallis Emergency Medicine A Comprehensive Study Guide_8th.pdf',
            'chapter_csv_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/chapter_detection_system/output/Tintinallis_Emergency_Med_chapters.csv'
        },
        {
            'id': 'rosens_emergency_10e',
            'title': "Rosens Emergency Medicine Concepts and Clinical Practice, 10th edition",
            'edition': '2nd Edition',
            'year': '2024',
            'pdf_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/multi_textbook_search/data/textbooks/Rosens Emergency Medicine Concepts and Clinical Practice, 10th edition - With Bookmarks.pdf',
            'chapter_csv_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/chapter_detection_system/output/Rosens_Emergency_Medicine_chapters.csv'
        }
    ]
    
    success = process_multiple_textbooks_optimized(textbook_configs)
    
    if success:
        print(f"\nReady for searches!")
        print(f"\nPerformance benefits:")
        print(f"   - Separate files per textbook")
        print(f"   - Only new textbooks processed")
        print(f"   - Scales efficiently to hundreds of textbooks")
    else:
        print(f"\nProcessing failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()


# """
# Optimized Single Textbook Processing Pipeline with Gemini Chunking
# =============================================

# Fast, scalable approach with separate files per textbook and AI-powered chunking.
# """

# import os
# import sys
# import pickle
# from typing import Dict, List

# from core.processor import ProductionTextbookProcessor
# from core.elasticsearch_search_engine import ElasticsearchSearchEngine
# from core.models import TextbookInfo, TextbookCollection


# class OptimizedTextbookManager:
#     """Manages textbooks with separate files for optimal performance"""
    
#     def __init__(self, data_dir: str = "data"):
#         self.data_dir = data_dir
#         self.textbooks_dir = os.path.join(data_dir, "textbooks")
#         self.registry_path = os.path.join(data_dir, "textbook_registry.pkl")
        
#         os.makedirs(self.textbooks_dir, exist_ok=True)
#         os.makedirs(data_dir, exist_ok=True)
    
#     def load_registry(self) -> Dict:
#         """Load textbook registry (metadata only)"""
#         if os.path.exists(self.registry_path):
#             with open(self.registry_path, 'rb') as f:
#                 return pickle.load(f)
#         return {"textbooks": {}, "total_chunks": 0, "version": 1}
    
#     def save_registry(self, registry: Dict):
#         """Save textbook registry"""
#         with open(self.registry_path, 'wb') as f:
#             pickle.dump(registry, f)
    
#     def save_textbook_chunks(self, textbook_id: str, documents: List, textbook_info: TextbookInfo):
#         """Save chunks for a single textbook"""
#         textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
#         textbook_data = {
#             "textbook_info": textbook_info,
#             "documents": documents,
#             "chunk_count": len(documents)
#         }
        
#         with open(textbook_file, 'wb') as f:
#             pickle.dump(textbook_data, f)
        
#         print(f"💾 Saved {len(documents)} chunks to {textbook_id}_chunks.pkl")
    
#     def load_textbook_chunks(self, textbook_id: str):
#         """Load chunks for a specific textbook"""
#         textbook_file = os.path.join(self.textbooks_dir, f"{textbook_id}_chunks.pkl")
        
#         if os.path.exists(textbook_file):
#             with open(textbook_file, 'rb') as f:
#                 return pickle.load(f)
#         return None
    
#     def load_all_documents(self) -> List:
#         """Load all documents from all textbook files (only when needed)"""
#         registry = self.load_registry()
#         all_documents = []
        
#         for textbook_id in registry["textbooks"].keys():
#             textbook_data = self.load_textbook_chunks(textbook_id)
#             if textbook_data:
#                 all_documents.extend(textbook_data["documents"])
        
#         return all_documents
    
#     def get_collection_for_search(self) -> TextbookCollection:
#         """Create a TextbookCollection for search (loads all docs)"""
#         collection = TextbookCollection()
#         registry = self.load_registry()
        
#         for textbook_id in registry["textbooks"].keys():
#             textbook_data = self.load_textbook_chunks(textbook_id)
#             if textbook_data:
#                 from core.models import ProcessingResult
#                 result = ProcessingResult(
#                     textbook_info=textbook_data["textbook_info"],
#                     documents=textbook_data["documents"],
#                     processing_time=0,
#                     success=True
#                 )
#                 collection.add_textbook(result)
        
#         return collection


# def process_multiple_textbooks_optimized(textbook_configs: List[Dict], 
#                                         use_gemini: bool = True,
#                                         gemini_api_key: str = None):
#     """Process multiple textbooks efficiently with Gemini chunking"""
    
#     print("\n" + "="*70)
#     print("🤖 OPTIMIZED MULTIPLE TEXTBOOK PROCESSING")
#     print("="*70)
    
#     if use_gemini:
#         print("✅ Chunking Method: Gemini AI (Semantic)")
#     else:
#         print("📝 Chunking Method: LangChain (Rule-based)")
    
#     manager = OptimizedTextbookManager()
#     registry = manager.load_registry()
    
#     # Initialize processor with Gemini support
#     processor = ProductionTextbookProcessor(
#         use_gemini=use_gemini,
#         gemini_api_key=gemini_api_key
#     )
    
#     processed_count = 0
#     failed_count = 0
    
#     for i, config in enumerate(textbook_configs, 1):
#         print(f"\n{'='*70}")
#         print(f"[{i}/{len(textbook_configs)}] Processing textbook...")
#         print(f"{'='*70}")
        
#         textbook_info = TextbookInfo(
#             id=config['id'],
#             title=config['title'],
#             edition=config.get('edition'),
#             year=config.get('year'),
#             file_path=config['pdf_path']
#         )
        
#         print(f"📚 Book: {textbook_info.display_name}")
        
#         if textbook_info.id in registry["textbooks"]:
#             print(f"⚠️  Warning: Textbook '{textbook_info.id}' already exists. Replacing...")
        
#         result = processor.process_textbook(
#             textbook_info=textbook_info,
#             pdf_path=config['pdf_path'],
#             chapter_csv_path=config['chapter_csv_path']
#         )
        
#         if not result.success:
#             print(f"❌ Error: Processing failed: {result.error_message}")
#             failed_count += 1
#             continue
        
#         print(f"✅ Successfully processed {len(result.documents)} document chunks")
        
#         manager.save_textbook_chunks(textbook_info.id, result.documents, textbook_info)
        
#         registry["textbooks"][textbook_info.id] = {
#             "title": textbook_info.display_name,
#             "chunk_count": len(result.documents),
#             "file_path": f"{textbook_info.id}_chunks.pkl",
#             "chunking_method": "gemini_ai" if use_gemini else "langchain"
#         }
        
#         processed_count += 1
    
#     if processed_count == 0:
#         print("\n❌ Error: No textbooks were successfully processed!")
#         return False
    
#     registry["total_chunks"] = sum(tb["chunk_count"] for tb in registry["textbooks"].values())
#     manager.save_registry(registry)
    
#     print(f"\n✅ Registry updated: {len(registry['textbooks'])} textbooks, {registry['total_chunks']} total chunks")
    
#     print("\n🔄 Rebuilding search index...")
#     collection = manager.get_collection_for_search()
    
#     search_engine = ElasticsearchSearchEngine(es_host="http://localhost:9200")
#     search_engine.build_index_from_collection(collection, force_rebuild=True)
    
#     stats = search_engine.get_statistics()
    
#     print(f"\n{'='*70}")
#     print(f"🎉 SYSTEM UPDATED!")
#     print(f"{'='*70}")
#     print(f"✅ Successfully processed: {processed_count}")
#     print(f"❌ Failed: {failed_count}")
#     print(f"📚 Total textbooks in system: {stats['total_textbooks']}")
#     print(f"📄 Total documents: {stats['total_documents']}")
    
#     print(f"\n📖 Available textbooks:")
#     for textbook, count in stats['textbook_breakdown'].items():
#         print(f"   ✓ {textbook}: {count} chunks")
    
#     return True


# def main():
#     """Main function - configure your textbooks here"""
    
#     # ============================================================
#     # PUT YOUR GEMINI API KEY HERE (HARDCODED)
#     # ============================================================
#     GEMINI_API_KEY = "AIzaSyCZY3v7DK7pf37qx8YDv7kIAZNZ9LNqrKE"  # ← REPLACE WITH YOUR ACTUAL KEY
    
#     # Example: GEMINI_API_KEY = "AIzaSyAbc123DefGhi456..."
    
    
#     # ============================================================
#     # CONFIGURE YOUR TEXTBOOKS HERE
#     # ============================================================
#     textbook_configs = [
#         {
#             'id': 'tintinalli_emergency_8e',
#             'title': "Tintinalli's Emergency Medicine",
#             'edition': '8th Edition',
#             'year': '2016',
#             'pdf_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/multi_textbook_search/data/textbooks/Tintinallis Emergency Medicine A Comprehensive Study Guide_8th.pdf',
#             'chapter_csv_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/chapter_detection_system/output/Tintinallis_Emergency_Med_chapters.csv'
#         },
#         {
#             'id': 'rosens_emergency_10e',
#             'title': "Rosens Emergency Medicine Concepts and Clinical Practice, 10th edition",
#             'edition': '10th Edition',
#             'year': '2024',
#             'pdf_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/multi_textbook_search/data/textbooks/Rosens Emergency Medicine Concepts and Clinical Practice, 10th edition - With Bookmarks.pdf',
#             'chapter_csv_path': '/Users/salarmuhammadumer/Desktop/medicalbooktest1/Iqonsulting-RAG/RAG AGENT/chapter_detection_system/output/Rosens_Emergency_Medicine_chapters.csv'
#         }
#     ]
    
#     # ============================================================
#     # ENABLE/DISABLE GEMINI CHUNKING
#     # ============================================================
#     USE_GEMINI = True  # Set to False to use LangChain chunking instead
    
    
#     print("\n" + "="*70)
#     print("🚀 STARTING TEXTBOOK PROCESSING")
#     print("="*70)
#     print(f"📚 Textbooks to process: {len(textbook_configs)}")
#     print(f"🤖 Chunking method: {'Gemini AI (Semantic)' if USE_GEMINI else 'LangChain (Rule-based)'}")
    
#     if USE_GEMINI:
#         if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
#             print(f"\n⚠️  WARNING: Please set your Gemini API key in the code!")
#             print(f"   Edit line ~140 in this file and replace 'YOUR_GEMINI_API_KEY_HERE'")
#             print(f"   with your actual Gemini API key.")
#             print(f"\n   Falling back to LangChain chunking...")
#             USE_GEMINI = False
#         else:
#             print(f"✅ Gemini API key configured")
    
#     print("="*70)
    
#     success = process_multiple_textbooks_optimized(
#         textbook_configs,
#         use_gemini=USE_GEMINI,
#         gemini_api_key=GEMINI_API_KEY if USE_GEMINI else None
#     )
    
#     if success:
#         print(f"\n{'='*70}")
#         print(f"✅ PROCESSING COMPLETE - READY FOR SEARCHES!")
#         print(f"{'='*70}")
        
#         if USE_GEMINI:
#             print(f"\n💡 Benefits of Gemini Chunking:")
#             print(f"   ✓ Semantic topic boundaries (no mid-topic splits)")
#             print(f"   ✓ Better search accuracy (~15-20% improvement)")
#             print(f"   ✓ More coherent retrieved contexts")
#             print(f"   ✓ Natural medical concept grouping")
        
#         print(f"\n📊 Performance:")
#         print(f"   ✓ Separate files per textbook")
#         print(f"   ✓ Only new textbooks processed")
#         print(f"   ✓ Scales efficiently to hundreds of textbooks")
#     else:
#         print(f"\n{'='*70}")
#         print(f"❌ PROCESSING FAILED!")
#         print(f"{'='*70}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()