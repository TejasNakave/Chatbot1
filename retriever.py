import numpy as np
import os
import pickle
import hashlib
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimpleRetriever:
    """
    A document retrieval system using TF-IDF and cosine similarity with caching.
    """
    
    def __init__(self, documents: List[Dict[str, str]], use_cache: bool = True):
        """
        Initialize the retriever with documents.
        
        Args:
            documents (List[Dict[str, str]]): List of document dictionaries
            use_cache (bool): Whether to use caching for TF-IDF vectors
        """
        self.documents = documents
        self.use_cache = use_cache
        self.cache_dir = "cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.doc_vectors = None
        self.build_index()
    
    def get_cache_key(self, documents: List[Dict[str, str]]) -> str:
        """Generate a cache key based on document content."""
        content_hash = hashlib.md5()
        for doc in documents:
            content_hash.update(doc.get('content', '').encode('utf-8'))
        return content_hash.hexdigest()[:12]
    
    def load_from_cache(self, cache_key: str) -> Tuple[bool, any, any]:
        """Load cached vectorizer and vectors."""
        try:
            vectorizer_path = os.path.join(self.cache_dir, f"vectorizer_{cache_key}.pkl")
            vectors_path = os.path.join(self.cache_dir, f"vectors_{cache_key}.pkl")
            documents_path = os.path.join(self.cache_dir, f"documents_{cache_key}.pkl")
            
            if all(os.path.exists(p) for p in [vectorizer_path, vectors_path, documents_path]):
                with open(vectorizer_path, 'rb') as f:
                    vectorizer = pickle.load(f)
                with open(vectors_path, 'rb') as f:
                    vectors = pickle.load(f)
                with open(documents_path, 'rb') as f:
                    cached_docs = pickle.load(f)
                
                return True, vectorizer, vectors
        except Exception as e:
            print(f"Cache load error: {e}")
        return False, None, None
    
    def save_to_cache(self, cache_key: str, vectorizer, vectors):
        """Save vectorizer and vectors to cache."""
        try:
            vectorizer_path = os.path.join(self.cache_dir, f"vectorizer_{cache_key}.pkl")
            vectors_path = os.path.join(self.cache_dir, f"vectors_{cache_key}.pkl")
            documents_path = os.path.join(self.cache_dir, f"documents_{cache_key}.pkl")
            
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(vectorizer, f)
            with open(vectors_path, 'wb') as f:
                pickle.dump(vectors, f)
            with open(documents_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            print(f"💾 Cached TF-IDF data with hash: {cache_key}...")
        except Exception as e:
            print(f"Cache save error: {e}")

    def build_index(self):
        """Build TF-IDF index for documents with caching support."""
        if not self.documents:
            print("No documents to index.")
            return
        
        # Try to load from cache first
        if self.use_cache:
            cache_key = self.get_cache_key(self.documents)
            loaded, vectorizer, vectors = self.load_from_cache(cache_key)
            if loaded:
                self.vectorizer = vectorizer
                self.doc_vectors = vectors
                print(f"📥 Loaded cached TF-IDF data: {cache_key}...")
                print(f"📥 Loaded cached index for {len(self.documents)} documents")
                return
        
        # Build index if not cached
        print(f"🔍 Building search index for {len(self.documents)} documents...")
        
        # Extract content from documents
        doc_contents = [doc.get('content', '') for doc in self.documents]
        
        # Create TF-IDF vectors
        self.doc_vectors = self.vectorizer.fit_transform(doc_contents)
        
        # Save to cache
        if self.use_cache:
            cache_key = self.get_cache_key(self.documents)
            self.save_to_cache(cache_key, self.vectorizer, self.doc_vectors)
        
        print(f"Built search index for {len(self.documents)} documents.")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Retrieve the most relevant document chunks for a given query.
        Uses both TF-IDF similarity and exact keyword matching for better results.
        Implements fallback strategies for better recall on question-based queries.
        
        Args:
            query (str): The search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            List[Dict[str, str]]: Most relevant documents
        """
        if self.doc_vectors is None or not self.documents:
            return []

        try:
            # TOPIC MISMATCH DETECTION - Check if query is about different domain
            query_lower = query.lower()
            
            # Define trade/business terms that our documents are about (excluding generic words)
            trade_terms = ['trade', 'export', 'import', 'customs', 'duty', 'tariff', 'svb', 
                          'business', 'commerce', 'shipping', 'freight', 'documentation', 'fta', 'drawback',
                          'compliance', 'regulation', 'clearance', 'invoice', 'certificate']
            
            # Define terms that indicate completely different topics
            unrelated_topics = ['coffee', 'cooking', 'recipe', 'food', 'kitchen', 'ingredient', 'baking',
                              'personal', 'family', 'health', 'medical', 'sports', 'entertainment', 'music',
                              'movie', 'game', 'hobby', 'travel', 'vacation', 'weather', 'animal', 'pet']
            
            # Check if query contains unrelated terms but no trade terms
            has_unrelated = any(term in query_lower for term in unrelated_topics)
            has_trade_terms = any(term in query_lower for term in trade_terms)
            
            print(f"🔍 RETRIEVER CHECK: Query='{query}', has_unrelated={has_unrelated}, has_trade_terms={has_trade_terms}")
            
            # CONTEXTUAL REFERENCE DETECTION - Check if query refers to previous context
            contextual_terms = ['above', 'previous', 'that', 'this', 'those', 'these', 'the image', 'the process', 'earlier']
            has_contextual_reference = any(term in query_lower for term in contextual_terms)
            
            if has_contextual_reference:
                print(f"🔗 CONTEXTUAL QUERY DETECTED: Query references previous context")
                print(f"🔗 Returning empty list to use conversation context instead of search")
                return []  # Let the AI use conversation context instead
            
            # Special handling for coffee queries - allow coffee content even from mixed documents
            if 'coffee' in query_lower:
                print(f"☕ COFFEE QUERY DETECTED: Looking for coffee-specific content")
                coffee_docs = []
                
                for doc in self.documents:
                    content_lower = doc.get('content', '').lower()
                    file_name = doc.get('file_name', '')
                    
                    # Include documents that contain coffee content
                    if 'coffee' in content_lower:
                        doc_copy = dict(doc)
                        # If it's an SVB document with coffee, mark it as coffee-focused
                        if 'SVB' in file_name.upper():
                            doc_copy['similarity_score'] = 0.90
                            doc_copy['match_type'] = 'coffee_from_svb'
                            print(f"☕ Found coffee content in SVB document: {doc.get('file_name', '')}")
                        else:
                            doc_copy['similarity_score'] = 0.95  # Higher for pure coffee docs
                            doc_copy['match_type'] = 'coffee_specific'
                            print(f"☕ Found pure coffee document: {doc.get('file_name', '')}")
                        coffee_docs.append(doc_copy)
                
                if coffee_docs:
                    print(f"☕ COFFEE RESULTS: Returning {len(coffee_docs)} documents with coffee content")
                    return coffee_docs[:top_k]
                else:
                    print(f"☕ NO COFFEE CONTENT FOUND: No documents contain coffee information")
                    return []
            
            # Special handling for SVB queries - focus on trade-related SVB content
            if 'svb' in query_lower and not 'coffee' in query_lower:
                print(f"🏛️ SVB QUERY DETECTED: Looking for SVB process content only")
                svb_docs = []
                
                for doc in self.documents:
                    content_lower = doc.get('content', '').lower()
                    file_name = doc.get('file_name', '')
                    
                    # For SVB queries, focus on SVB documents but prefer trade-related content
                    if 'SVB' in file_name.upper():
                        doc_copy = dict(doc)
                        # Check if content has both SVB and coffee - prefer SVB parts
                        if 'svb' in content_lower and 'special valuation' in content_lower:
                            doc_copy['similarity_score'] = 0.95  # High relevance for SVB match
                            doc_copy['match_type'] = 'svb_specific'
                        else:
                            doc_copy['similarity_score'] = 0.85  # Lower for mixed content
                            doc_copy['match_type'] = 'svb_mixed'
                        svb_docs.append(doc_copy)
                        print(f"🏛️ Found SVB document: {doc.get('file_name', '')}")
                
                if svb_docs:
                    print(f"🏛️ SVB RESULTS: Returning {len(svb_docs)} SVB-specific documents")
                    return svb_docs[:top_k]
            
            if has_unrelated and not has_trade_terms:
                print(f"🚫 TOPIC MISMATCH: Query about '{query}' is unrelated to trade documents")
                return []  # Return empty list to trigger global search

            # PRIORITY KEYWORD MATCHING - Force specific documents for flowchart queries
            query_lower = query.lower()
            forced_results = []
            
            print(f"🔍 Query: '{query}' -> Checking for specific flowcharts...")
            
            # Bill of Entry (BOE) - highest priority for BOE keywords
            if any(keyword in query_lower for keyword in ['bill of entry', 'boe', 'bill entry', 'entry bill']):
                print("🎯 Detected BILL OF ENTRY (BOE) query")
                for doc in self.documents:
                    content_lower = doc.get('content', '').lower()
                    file_name_lower = doc.get('file_name', '').lower()
                    # Look for BOE content in any document
                    if any(boe_term in content_lower for boe_term in ['bill of entry', 'boe', '1105807', 'customs edi']):
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_boe'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')} (contains BOE)")
                        break
                    # Also check filename for BOE-related terms
                    elif any(boe_term in file_name_lower for boe_term in ['bill of entry', 'boe', 'import', 'customs']):
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.95
                        doc_copy['match_type'] = 'forced_boe_filename'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')} (filename match)")
                        break
            
            # Duty Drawback Flowchart - highest priority for duty/drawback keywords
            elif any(keyword in query_lower for keyword in ['duty drawback', 'drawback flowchart', 'duty flow']):
                print("🎯 Detected DUTY DRAWBACK query")
                for doc in self.documents:
                    file_name_lower = doc.get('file_name', '').lower()
                    if 'duty drawback flowchart' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_duty_drawback'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')}")
                        break
            
            # Export Process Flowchart
            elif any(keyword in query_lower for keyword in ['export process', 'export flowchart', 'export flow']):
                print("🎯 Detected EXPORT PROCESS query")
                for doc in self.documents:
                    file_name_lower = doc.get('file_name', '').lower()
                    if 'export process flowchart' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_export'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')}")
                        break
            
            # Import Process Flowchart
            elif any(keyword in query_lower for keyword in ['import process', 'import flowchart', 'import flow']):
                print("🎯 Detected IMPORT PROCESS query")
                for doc in self.documents:
                    file_name_lower = doc.get('file_name', '').lower()
                    if 'import process flowchart' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_import'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')}")
                        break
            
            # FTA Flowchart
            elif any(keyword in query_lower for keyword in ['fta', 'free trade', 'fta flowchart']):
                print("🎯 Detected FTA query")
                for doc in self.documents:
                    file_name_lower = doc.get('file_name', '').lower()
                    if 'fta flowchart' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_fta'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')}")
                        break
            
            # Customs Tariff Flowchart - NEW!
            elif any(keyword in query_lower for keyword in ['customs tariff', 'tariff flowchart', 'customs flowchart', 'hs code']):
                print("🎯 Detected CUSTOMS TARIFF query")
                for doc in self.documents:
                    file_name_lower = doc.get('file_name', '').lower()
                    # Look for the specific customs tariff PDF with flowcharts
                    if 'customs tarriff & hs code' in file_name_lower and '.pdf' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.99
                        doc_copy['match_type'] = 'forced_customs_tariff'
                        forced_results.append(doc_copy)
                        print(f"   ✅ FORCED MATCH: {doc.get('file_name', '')} (PDF with images)")
                        break
                    # Backup: also look for other customs tariff content
                    elif 'customs tariff' in file_name_lower or 'customs tarriff' in file_name_lower:
                        doc_copy = dict(doc)
                        doc_copy['similarity_score'] = 0.95
                        doc_copy['match_type'] = 'forced_customs_backup'
                        forced_results.append(doc_copy)
                        print(f"   ✅ BACKUP MATCH: {doc.get('file_name', '')}")
            
            # If we found forced results, return them immediately
            if forced_results:
                print(f"🎯 FORCED MATCH SUCCESS: Returning {len(forced_results)} documents")
                return forced_results[:top_k]

            # Extract key terms from questions for better search
            def extract_key_terms(query_text):
                """Extract meaningful terms from questions, removing stop words and question words."""
                import re
                # Remove common question words and patterns
                question_words = ['what', 'is', 'are', 'how', 'when', 'where', 'why', 'who', 'which', 'the', 'a', 'an']
                
                # Clean the query
                cleaned = re.sub(r'[^\w\s]', ' ', query_text.lower())
                words = cleaned.split()
                
                # Remove question words and short words
                key_terms = [word for word in words if word not in question_words and len(word) > 2]
                
                return ' '.join(key_terms)

            # Try multiple search strategies
            all_results = []
            
            # Strategy 1: Original query
            results1 = self._search_with_query(query, top_k)
            all_results.extend(results1)
            
            # Strategy 2: If no results from original query, try key terms only
            if not results1:
                key_terms = extract_key_terms(query)
                if key_terms and key_terms != query.lower():
                    results2 = self._search_with_query(key_terms, top_k)
                    all_results.extend(results2)
            
            # Strategy 3: If still no results, try individual significant words
            if not all_results:
                key_terms = extract_key_terms(query)
                words = key_terms.split()
                for word in words:
                    if len(word) > 4:  # Only try longer, more significant words
                        word_results = self._search_with_query(word, top_k)
                        all_results.extend(word_results)
                        if word_results:  # If we find something, stop searching
                            break
            
            # Remove duplicates and sort by similarity
            seen = set()
            unique_results = []
            for doc in all_results:
                doc_id = (doc['file_name'], doc.get('content', '')[:100])  # Use filename + content snippet as ID
                if doc_id not in seen:
                    seen.add(doc_id)
                    unique_results.append(doc)
            
            # Sort by similarity score and return top results
            unique_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            return unique_results[:top_k]
            
        except Exception as e:
            print(f"Error in retrieval: {e}")
            return []

    def _search_with_query(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        """
        Internal method to perform search with a specific query string.
        
        Args:
            query (str): The search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            List[Dict[str, str]]: Most relevant documents
        """
        try:
            query_lower = query.lower()
            
            # PRIORITY: Prioritize specific flowchart documents for certificate/process queries
            # This helps when comprehensive books compete with specific flowcharts
            if any(term in query_lower for term in ['certificate of origin', 'certificate', 'origin', 'bill of entry', 'boe']):
                print("📋 CERTIFICATE/BOE QUERY: Prioritizing flowchart documents over comprehensive books")
                
                # Find flowchart documents first
                flowchart_docs = []
                comprehensive_docs = []
                
                for doc in self.documents:
                    filename = doc.get('file_name', '').lower()
                    content = doc.get('content', '').lower()
                    
                    # Prioritize specific flowchart files and BOE-containing documents
                    if any(flowchart_term in filename for flowchart_term in [
                        'import process flowchart', 'export process flowchart', 
                        'duty drawback flowchart', 'fta flowchart']) or \
                       any(boe_term in content for boe_term in ['bill of entry', 'boe', '1105807']):
                        flowchart_docs.append(doc)
                        print(f"📋 PRIORITY: {doc.get('file_name', '')}")
                    # Deprioritize comprehensive books
                    elif any(book_term in filename for book_term in [
                        'impex new book', 'comprehensive', 'manual', 'guide']):
                        comprehensive_docs.append(doc)
                        print(f"📚 DEPRIORITIZED: {doc.get('file_name', '')}")
                    else:
                        flowchart_docs.append(doc)  # Treat other docs as medium priority
                
                # Use flowchart documents first, then add comprehensive if needed
                prioritized_documents = flowchart_docs + comprehensive_docs
                print(f"📋 Using {len(flowchart_docs)} priority docs + {len(comprehensive_docs)} comprehensive docs")
            else:
                prioritized_documents = self.documents
            
            # First, try exact keyword matching for better recall on specific terms
            exact_matches = []
            
            # Extract potential date/keyword patterns from query
            import re
            date_patterns = re.findall(r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b', query_lower)
            date_patterns.extend(re.findall(r'\b\d{1,2}[/-]\d{4}\b', query_lower))
            date_patterns.extend(re.findall(r'\b\d{4}[/-]\d{1,2}\b', query_lower))
            
            # Add the full query for exact matching
            search_terms = [query_lower] + date_patterns
            
            for i, doc in enumerate(prioritized_documents):
                content_lower = doc.get('content', '').lower()
                max_score = 0
                match_found = False
                
                for term in search_terms:
                    if term in content_lower:
                        match_found = True
                        # Score based on term length and specificity
                        score = min(1.0, len(term) / 20.0 + 0.5)
                        max_score = max(max_score, score)
                
                if match_found:
                    doc_copy = dict(doc)
                    doc_copy['similarity_score'] = max_score
                    doc_copy['match_type'] = 'exact'
                    exact_matches.append(doc_copy)
            
            # Create TF-IDF vectors for prioritized documents only
            if hasattr(self, '_prioritized_vectors') and len(prioritized_documents) != len(self.documents):
                # Need to rebuild vectors for prioritized subset
                prioritized_contents = [doc.get('content', '') for doc in prioritized_documents]
                prioritized_vectors = self.vectorizer.transform(prioritized_contents)
            else:
                prioritized_vectors = self.doc_vectors
            
            # Vectorize the query for TF-IDF search
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, prioritized_vectors).flatten()
            
            # Get top-k most similar documents (limit to available documents)
            num_docs = len(prioritized_documents)
            actual_k = min(top_k, num_docs)
            top_indices = np.argsort(similarities)[::-1][:actual_k]
            
            # Get TF-IDF matches
            tfidf_matches = []
            for i in range(len(top_indices)):
                idx = int(top_indices[i])  # Convert to Python int
                similarity_score = float(similarities[idx])  # Convert to Python float
                
                # Apply minimum similarity threshold (lowered for better recall)
                if similarity_score > 0.05:
                    doc = dict(prioritized_documents[idx])  # Create a new dict
                    doc['similarity_score'] = similarity_score
                    doc['match_type'] = 'tfidf'
                    tfidf_matches.append(doc)
                    tfidf_matches.append(doc)
            
            # Combine and deduplicate results
            all_matches = {}
            
            # Add exact matches first (higher priority)
            for doc in exact_matches:
                file_name = doc['file_name']
                all_matches[file_name] = doc
            
            # Add TF-IDF matches if not already included
            for doc in tfidf_matches:
                file_name = doc['file_name']
                if file_name not in all_matches:
                    all_matches[file_name] = doc
                elif all_matches[file_name]['similarity_score'] < doc['similarity_score']:
                    # Keep the higher score but mark as hybrid
                    all_matches[file_name]['similarity_score'] = max(
                        all_matches[file_name]['similarity_score'], 
                        doc['similarity_score']
                    )
                    all_matches[file_name]['match_type'] = 'hybrid'
            
            # Sort by similarity score and return top results
            relevant_docs = list(all_matches.values())
            relevant_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return relevant_docs[:top_k]
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            date_patterns.extend(re.findall(r'\b\d{4}[/-]\d{1,2}\b', query_lower))
            
            # Add the full query for exact matching
            search_terms = [query_lower] + date_patterns
            
            for i, doc in enumerate(self.documents):
                content_lower = doc.get('content', '').lower()
                max_score = 0
                match_found = False
                
                for term in search_terms:
                    if term in content_lower:
                        match_found = True
                        # Score based on term length and specificity
                        score = min(1.0, len(term) / 20.0 + 0.5)
                        max_score = max(max_score, score)
                
                if match_found:
                    doc_copy = dict(doc)
                    doc_copy['similarity_score'] = max_score
                    doc_copy['match_type'] = 'exact'
                    exact_matches.append(doc_copy)
            
            # Vectorize the query for TF-IDF search
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarity scores
            similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
            
            # Get top-k most similar documents (limit to available documents)
            num_docs = len(self.documents)
            actual_k = min(top_k, num_docs)
            top_indices = np.argsort(similarities)[::-1][:actual_k]
            
            # Get TF-IDF matches
            tfidf_matches = []
            for i in range(len(top_indices)):
                idx = int(top_indices[i])  # Convert to Python int
                similarity_score = float(similarities[idx])  # Convert to Python float
                
                # Apply minimum similarity threshold (lowered for better recall)
                if similarity_score > 0.05:
                    doc = dict(self.documents[idx])  # Create a new dict
                    doc['similarity_score'] = similarity_score
                    doc['match_type'] = 'tfidf'
                    tfidf_matches.append(doc)
            
            # Combine and deduplicate results
            all_matches = {}
            
            # Add exact matches first (higher priority)
            for doc in exact_matches:
                file_name = doc['file_name']
                all_matches[file_name] = doc
            
            # Add TF-IDF matches if not already included
            for doc in tfidf_matches:
                file_name = doc['file_name']
                if file_name not in all_matches:
                    all_matches[file_name] = doc
                elif all_matches[file_name]['similarity_score'] < doc['similarity_score']:
                    # Keep the higher score but mark as hybrid
                    all_matches[file_name]['similarity_score'] = max(
                        all_matches[file_name]['similarity_score'], 
                        doc['similarity_score']
                    )
                    all_matches[file_name]['match_type'] = 'hybrid'
            
            # Sort by similarity score and return top results
            relevant_docs = list(all_matches.values())
            relevant_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return relevant_docs[:top_k]
            
        except Exception as e:
            print(f"Error during retrieval: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def search_documents(self, query: str, top_k: int = 3) -> str:
        """
        Search documents and return formatted context string.
        
        Args:
            query (str): The search query
            top_k (int): Number of top documents to retrieve
            
        Returns:
            str: Formatted context string
        """
        relevant_docs = self.retrieve_relevant_chunks(query, top_k)
        
        if not relevant_docs:
            return "No relevant documents found."
        
        context = ""
        for i, doc in enumerate(relevant_docs):
            score = doc.get('similarity_score', 0.0)
            file_name = doc.get('file_name', 'Unknown')
            content = doc.get('content', '')
            
            context += f"Relevant Document {i+1} - {file_name} (Score: {score:.3f}):\n"
            
            # Limit content length
            if len(content) > 1000:
                context += content[:1000] + "...\n\n"
            else:
                context += content + "\n\n"
        
        return context


if __name__ == "__main__":
    # Test the retriever with sample documents
    sample_docs = [
        {"file_name": "doc1.txt", "content": "Python is a programming language used for web development."},
        {"file_name": "doc2.txt", "content": "Machine learning is a subset of artificial intelligence."},
        {"file_name": "doc3.txt", "content": "Web development involves creating websites and web applications."}
    ]
    
    retriever = SimpleRetriever(sample_docs)
    
    query = "What is Python used for?"
    relevant = retriever.retrieve_relevant_chunks(query)
    
    print(f"Query: {query}")
    print(f"Found {len(relevant)} relevant documents:")
    for doc in relevant:
        print(f"- {doc['file_name']} (Score: {doc['similarity_score']:.3f})")
