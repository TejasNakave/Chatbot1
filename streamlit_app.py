import streamlit as st
import os
import sys
import re
from datetime import datetime
from document_loader import load_documents_from_folder
from gemini_wrapper import GeminiAPIWrapper
from retriever import SimpleRetriever


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_documents():
    """Load documents with caching and better error handling."""
    try:
        # Ensure data directory exists
        data_dir = "data/"
        if not os.path.exists(data_dir):
            st.error(f"Data directory '{data_dir}' not found. Please ensure it exists on the server.")
            return []
        
        documents = load_documents_from_folder(data_dir)
        st.success(f"✅ Loaded {len(documents)} documents successfully")
        return documents
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")
        return []


def refresh_documents():
    """Force refresh of documents by clearing cache."""
    # Clear all caches to ensure fresh reload
    load_documents.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    
    # Force reload documents from folder
    return load_documents_from_folder("data/")


@st.cache_resource(show_spinner=False)
def initialize_components(documents):
    """Initialize chatbot components with caching and user isolation."""
    try:
        # Add user session isolation
        session_id = st.session_state.get('session_id', None)
        if not session_id:
            import time
            session_id = f"session_{int(time.time())}"
            st.session_state.session_id = session_id
        
        retriever = SimpleRetriever(documents)
        gemini = GeminiAPIWrapper()
        
        # Log initialization for debugging
        print(f"🚀 Initialized components for {session_id}")
        return retriever, gemini, None
    except Exception as e:
        print(f"❌ Error initializing components: {str(e)}")
        return None, None, str(e)


def format_chat_message(role, content, timestamp=None):
    """Format a chat message for display with intelligent structure and image support."""
    
    if role == "user":
        st.chat_message("user").write(content)
    else:
        # Check for images in the content
        import re
        
        # Extract image references
        image_pattern = r'\[IMAGE_\d+: ([^\]]+)\]'
        images = re.findall(image_pattern, content)
        
        # Debug: Print what we found
        if images:
            print(f"🖼️ Found {len(images)} images in content:")
            for i, img_path in enumerate(images):
                print(f"  Image {i}: {img_path}")
                print(f"  Exists: {os.path.exists(img_path)}")
        else:
            # Check if content contains image references at all
            if 'IMAGE_' in content:
                print(f"🔍 Content contains IMAGE_ but no matches found:")
                print(f"Pattern: {image_pattern}")
                # Show first 500 chars of content for debugging
                print(f"Content preview: {content[:500]}")
        
        # Remove image references from text content for processing
        clean_content = re.sub(image_pattern, '', content)
        clean_content = re.sub(r'\[IMAGES_AVAILABLE: \d+ images extracted\]', '', clean_content)
        clean_content = clean_content.strip()
        
        # Smart formatting based on content structure
        if isinstance(clean_content, str) and clean_content:
            # Check if content already has proper bullet structure
            has_bullets = bool(re.search(r'^\s*[-•*]\s+', clean_content, re.MULTILINE))
            has_numbers = bool(re.search(r'^\s*\d+\.\s+', clean_content, re.MULTILINE))
            
            if has_bullets or has_numbers:
                # Content already has proper bullets/numbers - display as is
                st.chat_message("assistant").markdown(clean_content)
            else:
                # Content needs formatting - but PRIORITIZE continuous text
                if '\n' in clean_content and len(clean_content.split('\n')) > 2:
                    # Multi-line content - ALWAYS join into continuous text, NEVER create bullets
                    lines = [line.strip() for line in clean_content.split('\n') if line.strip()]
                    
                    # Ultra-aggressive joining - join ALL lines into continuous text
                    # Only separate at clear sentence boundaries
                    joined_text = []
                    current_sentence = ""
                    
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # If current sentence is empty, start new sentence
                        if not current_sentence:
                            current_sentence = line
                        else:
                            # Always join unless the previous line clearly ends a complete sentence
                            # AND the new line clearly starts a new complete sentence
                            prev_ends_sentence = current_sentence.rstrip().endswith(('.', '!', '?', ':'))
                            new_starts_sentence = line[0].isupper() and len(line.split()) > 3
                            
                            # Even if both conditions are met, still join if it looks like continuation
                            first_word = line.split()[0].lower() if line.split() else ""
                            is_continuation = first_word in ['the', 'this', 'customs', 'exporter', 'payment', 'goods', 'documents', 'bank', 'process', 'once', 'after', 'before', 'when', 'then', 'finally', 'an', 'and']
                            
                            if prev_ends_sentence and new_starts_sentence and not is_continuation:
                                # Start new sentence
                                joined_text.append(current_sentence)
                                current_sentence = line
                            else:
                                # Join to current sentence
                                current_sentence += ' ' + line
                    
                    # Add the last sentence
                    if current_sentence:
                        joined_text.append(current_sentence)
                    
                    # NEVER create bullets - always display as continuous text
                    continuous_text = ' '.join(joined_text)
                    st.chat_message("assistant").write(continuous_text)
                elif ';' in clean_content and len(clean_content.split(';')) > 5:
                    # Only create bullets for semicolon lists if there are MANY items (5+)
                    items = [item.strip() for item in clean_content.split(';') if item.strip()]
                    bullets = '\n'.join([f'- {item}' for item in items])
                    st.chat_message("assistant").markdown(bullets)
                else:
                    # Default: Always display as continuous text
                    st.chat_message("assistant").write(clean_content)
        
        # Display images if any were found
        if images:
            print(f"📸 Displaying {len(images)} images in Streamlit interface")
            with st.chat_message("assistant"):
                st.markdown("**📸 Related Images:**")
                cols = st.columns(min(len(images), 3))  # Max 3 images per row
                
                displayed_images = 0
                for i, image_path in enumerate(images):
                    print(f"  Attempting to display image {i}: {image_path}")
                    with cols[i % 3]:
                        try:
                            # Fix path for hosted environments
                            if not os.path.isabs(image_path):
                                # Make sure we're using the correct relative path
                                if not image_path.startswith('cache/'):
                                    image_path = os.path.join('cache', os.path.basename(image_path))
                            
                            if os.path.exists(image_path):
                                print(f"    ✅ Image file exists, displaying...")
                                st.image(image_path, caption=f"Image {i+1}", use_container_width=True)
                                print(f"    ✅ Image displayed successfully")
                                displayed_images += 1
                            else:
                                print(f"    ❌ Image file not found: {image_path}")
                                # Don't show error to user, just log it
                                print(f"    📁 Current working directory: {os.getcwd()}")
                                print(f"    📁 Cache directory exists: {os.path.exists('cache')}")
                                if os.path.exists('cache'):
                                    print(f"    📁 Cache contents: {os.listdir('cache')[:5]}")  # Show first 5 files
                        except Exception as e:
                            print(f"    ❌ Error displaying image: {str(e)}")
                            # Don't show error to user in production
                            
                if displayed_images == 0:
                    st.info("📸 Images are available but not accessible in hosted environment. This is a known limitation.")
        
        # If no clean content and no images, show original content
        if not clean_content and not images:
            st.chat_message("assistant").write(content)


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Trade Assistant",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS and JS for better styling and image protection
    st.markdown('''
    <style>
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
    }
    /* Prevent drag and selection on images */
    img {
        -webkit-user-drag: none;
        user-drag: none;
        -webkit-user-select: none;
        user-select: none;
        pointer-events: auto;
    }
    </style>
    <script>
    // More robust image protection
    function protectAllImages() {
        document.querySelectorAll('img').forEach(function(img) {
            // Disable right-click
            img.addEventListener('contextmenu', function(e) { 
                e.preventDefault(); 
                e.stopPropagation();
                return false;
            });
            // Disable drag
            img.addEventListener('dragstart', function(e) { 
                e.preventDefault(); 
                e.stopPropagation();
                return false;
            });
            // Disable selection
            img.addEventListener('selectstart', function(e) { 
                e.preventDefault(); 
                return false;
            });
            // Additional protection
            img.style.webkitUserSelect = 'none';
            img.style.userSelect = 'none';
            img.style.webkitUserDrag = 'none';
            img.style.userDrag = 'none';
            img.setAttribute('draggable', 'false');
            img.setAttribute('unselectable', 'on');
        });
    }
    
    // Run on page load
    document.addEventListener("DOMContentLoaded", protectAllImages);
    
    // Run when new content is added (for chat messages)
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.addedNodes.length > 0) {
                setTimeout(protectAllImages, 100); // Small delay to ensure images are loaded
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Also disable right-click on the entire document for extra protection
    document.addEventListener('contextmenu', function(e) {
        if (e.target.tagName === 'IMG') {
            e.preventDefault();
            e.stopPropagation();
            return false;
        }
    });
    </script>
    ''', unsafe_allow_html=True)
    
    # Additional CSS for chat styling
    st.markdown("""
    <style>
    /* Increase chat message text size - clean version */
    .stChatMessage {
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
    }
    
    .stChatMessage p {
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .stChatMessage ul, .stChatMessage ol {
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
    }
    
    .stChatMessage li {
        font-size: 1.3rem !important;
        margin-bottom: 0.6rem !important;
    }
    
    /* Increase chat input text size */
    .stChatInput > div > div > input {
        font-size: 1.2rem !important;
        padding: 0.8rem !important;
    }
    
    /* Make chat messages more readable */
    [data-testid="stChatMessage"] {
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] div {
        font-size: 1.3rem !important;
        line-height: 1.7 !important;
    }
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background: #fafafa;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load documents initially and store in session state
    if 'documents' not in st.session_state or st.session_state.get('reload_documents', False):
        with st.spinner("Loading documents..."):
            st.session_state.documents = load_documents()
            st.session_state.reload_documents = False
            # Clear the initialized flag to force reinitialization of components
            if st.session_state.get('reload_documents', False):
                st.session_state.pop('initialized', None)
                st.session_state.pop('retriever', None)
                st.session_state.pop('gemini', None)
    
    documents = st.session_state.documents
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Set default values for removed sidebar controls
    max_docs = 3  # Default number of relevant documents
    show_sources = False  # Default to not showing sources
    
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing AI components..."):
            retriever, gemini, error = initialize_components(documents)
            if error:
                st.error(f"❌ Error initializing AI: {error}")
                st.info("Please check your .env file and ensure GEMINI_API_KEY is set correctly.")
                return
            st.session_state.retriever = retriever
            st.session_state.gemini = gemini
            st.session_state.initialized = True
    
    # Main chat interface with improved layout
    st.subheader("💬 Chat Interface")
    
    # Display chat history with enhanced styling
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for entry in st.session_state.chat_history:
                format_chat_message("user", entry["question"], entry["timestamp"])
                
                if show_sources and entry.get("sources"):
                    with st.expander("📄 Sources used:"):
                        for source in entry["sources"]:
                            relevance = source.get('similarity_score', 0)
                            st.write(f"- **{source['file_name']}** (relevance: {relevance:.3f})")
                
                format_chat_message("assistant", entry["answer"], entry["timestamp"])
                st.divider()
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Ask me anything about your documents.")
    
    # Chat input
    user_question = st.chat_input("Ask a question about your documents...")
    
    if user_question:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add user question to history immediately
        format_chat_message("user", user_question, timestamp)
        
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                # For competency/capability questions, use more documents to get complete info
                search_terms = user_question.lower()
                if any(term in search_terms for term in ['competenc', 'skill', 'service', 'capabilit', 'core', 'all']):
                    # Use more documents for comprehensive answers
                    retrieval_count = min(max_docs + 2, 10)
                else:
                    retrieval_count = max_docs
                
                # Retrieve relevant documents
                relevant_docs = st.session_state.retriever.retrieve_relevant_chunks(
                    user_question, top_k=retrieval_count
                )
                
                # Special handling for SVB queries - ensure SVB document is included
                if 'svb' in user_question.lower() or 'process flow chart' in user_question.lower():
                    # Find SVB document in all documents
                    svb_doc = None
                    for doc in st.session_state.documents:
                        if 'SVB' in doc.get('file_name', '').upper() or 'svb' in doc.get('file_name', '').lower():
                            svb_doc = doc.copy()  # Make a copy
                            svb_doc['similarity_score'] = 1.0  # Set high relevance
                            break
                    
                    # Add SVB document to relevant docs if found and not already included
                    if svb_doc:
                        # Check if already included
                        already_included = any(doc.get('file_name') == svb_doc.get('file_name') for doc in relevant_docs)
                        if not already_included:
                            relevant_docs.insert(0, svb_doc)  # Add at the beginning with highest priority
                            print(f"🎯 Added SVB document to context: {svb_doc.get('file_name')}")
                        else:
                            print(f"🎯 SVB document already in context: {svb_doc.get('file_name')}")
                    else:
                        print(f"⚠️ SVB document not found in loaded documents")
                
                # Show sources if enabled
                if show_sources and relevant_docs:
                    with st.expander("📄 Sources found:"):
                        for doc in relevant_docs:
                            st.write(f"- **{doc['file_name']}** (relevance: {doc['similarity_score']:.3f})")
                
                # Generate answer
                answer = st.session_state.gemini.chat_with_context(
                    user_question, 
                    relevant_docs, 
                    st.session_state.chat_history
                )
                
                # Display answer
                format_chat_message("assistant", answer, timestamp)
                
                # Save to session state
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": answer,
                    "sources": relevant_docs,
                    "timestamp": timestamp
                })
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
