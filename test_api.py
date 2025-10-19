"""
Test Client for Medical Textbook Retrieval API
==============================================
Simple script to test your Flask API
"""

import requests
import json
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:5000"


def print_separator():
    print("\n" + "="*70 + "\n")


def test_health_check():
    """Test if API is running"""
    print("🏥 Testing API Health Check...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data['status']}")
            print(f"📊 Statistics:")
            stats = data['statistics']
            print(f"   Total Documents: {stats['total_documents']}")
            print(f"   Total Textbooks: {stats['total_textbooks']}")
            print(f"   Embedding Model: {stats['embedding_model']}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")
        print("   Run: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_search(query: str, top_k: int = 5):
    """Test basic search functionality"""
    print(f"🔍 Searching for: '{query}'")
    print(f"   Retrieving top {top_k} results...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "top_k": top_k
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['num_results']} results\n")
            
            for result in data['results']:
                print(f"📖 Rank {result['rank']} (Score: {result['relevance_score']:.3f})")
                print(f"   Textbook: {result['source']['textbook']}")
                print(f"   Chapter: {result['source']['chapter']}")
                print(f"   Page: {result['source']['page']}")
                print(f"   Preview: {result['content_preview'][:150]}...")
                print()
            
            return data
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(response.json())
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_qa_search(question: str, answer: str = "", top_k: int = 5):
    """Test Q&A search functionality"""
    print(f"🔍 Q&A Search")
    print(f"   Question: {question}")
    if answer:
        print(f"   Answer: {answer}")
    
    try:
        payload = {
            "question": question,
            "top_k": top_k
        }
        if answer:
            payload["answer"] = answer
        
        response = requests.post(
            f"{API_BASE_URL}/search/qa",
            json=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ Found {data['num_sources']} relevant sources\n")
            
            if data['primary_source']:
                print("🎯 Primary Source:")
                primary = data['primary_source']
                print(f"   Textbook: {primary['source']['textbook']}")
                print(f"   Chapter: {primary['source']['chapter']}")
                print(f"   Page: {primary['source']['page']}")
                print(f"   Score: {primary['relevance_score']:.3f}")
                print()
            
            print("📚 All Sources:")
            for source in data['sources']:
                print(f"   [{source['rank']}] {source['source']['textbook']}")
                print(f"       Chapter: {source['source']['chapter']}, Page: {source['source']['page']}")
            
            return data
        else:
            print(f"❌ Q&A search failed: {response.status_code}")
            print(response.json())
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_list_textbooks():
    """Test listing available textbooks"""
    print("📚 Listing Available Textbooks...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/textbooks")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {data['total_textbooks']} textbooks\n")
            
            for textbook in data['textbooks']:
                print(f"   📖 {textbook['id']}")
                print(f"      Documents: {textbook['document_count']}")
            
            return data
        else:
            print(f"❌ Failed: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def test_get_stats():
    """Test getting system statistics"""
    print("📊 Getting System Statistics...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/stats")
        
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            
            print(f"✅ System Statistics:")
            print(f"   Total Documents: {stats['total_documents']}")
            print(f"   Total Textbooks: {stats['total_textbooks']}")
            print(f"   Embedding Dimension: {stats['embedding_dimension']}")
            print(f"   Model: {stats['model_name']}")
            print(f"\n   Textbook Breakdown:")
            for textbook, count in stats['textbook_breakdown'].items():
                print(f"      {textbook}: {count} documents")
            
            return data
        else:
            print(f"❌ Failed: {response.status_code}")
            return None
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def run_all_tests():
    """Run all API tests"""
    print("\n" + "🧪 MEDICAL TEXTBOOK API TEST SUITE ".center(70, "="))
    
    # Test 1: Health Check
    print_separator()
    if not test_health_check():
        print("\n❌ API is not running. Please start it first.")
        return
    
    # Test 2: List Textbooks
    print_separator()
    test_list_textbooks()
    
    # Test 3: Get Statistics
    print_separator()
    test_get_stats()
    
    # Test 4: Basic Search
    print_separator()
    test_search("What is tuberculosis?", top_k=3)
    
    # Test 5: Q&A Search
    print_separator()
    test_qa_search(
        question="What is the most common cause of esophageal rupture?",
        answer="Forceful vomiting",
        top_k=3
    )
    
    print_separator()
    print("✅ All tests completed!")
    print("="*70)


def interactive_mode():
    """Interactive mode for testing"""
    print("\n" + "🔬 INTERACTIVE API TEST MODE ".center(70, "="))
    print("\nAvailable commands:")
    print("  1. Health check")
    print("  2. List textbooks")
    print("  3. Get statistics")
    print("  4. Search query")
    print("  5. Q&A search")
    print("  6. Run all tests")
    print("  q. Quit")
    print("="*70)
    
    while True:
        print()
        choice = input("Enter command (1-6, or q to quit): ").strip()
        
        if choice == 'q':
            print("👋 Goodbye!")
            break
        elif choice == '1':
            print_separator()
            test_health_check()
        elif choice == '2':
            print_separator()
            test_list_textbooks()
        elif choice == '3':
            print_separator()
            test_get_stats()
        elif choice == '4':
            print_separator()
            query = input("Enter search query: ").strip()
            if query:
                top_k = input("Number of results (default 5): ").strip()
                top_k = int(top_k) if top_k else 5
                test_search(query, top_k)
        elif choice == '5':
            print_separator()
            question = input("Enter question: ").strip()
            answer = input("Enter answer (optional): ").strip()
            if question:
                top_k = input("Number of results (default 5): ").strip()
                top_k = int(top_k) if top_k else 5
                test_qa_search(question, answer, top_k)
        elif choice == '6':
            run_all_tests()
        else:
            print("❌ Invalid choice. Please enter 1-6 or q.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'auto':
        # Run all tests automatically
        run_all_tests()
    else:
        # Interactive mode
        interactive_mode()