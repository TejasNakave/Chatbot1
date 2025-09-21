import streamlit as st
import os
import sys
import re
from datetime import datetime
from document_loader import load_documents_from_folder
from gemini_wrapper import GeminiAPIWrapper
from retriever import SimpleRetriever


def get_suggested_questions():
    """Generate contextual suggested questions based on document content"""
    suggestions = [
        "💼 What is DGFT and what services do they provide?",
        "📋 How do I apply for an Import Export Code (IEC)?", 
        "🎯 What are the latest export incentive schemes?",
        "💰 How do duty drawbacks work in international trade?",
        "📊 What documents are required for export transactions?",
        "🌏 What are the current trade policies with major countries?",
        "🔍 How can I check the status of my trade license?",
        "📈 What are the benefits of SEZ (Special Economic Zone)?",
        "⚖️ What are the compliance requirements for exporters?",
        "🚢 How do I calculate shipping and logistics costs?"
    ]
    return suggestions

def generate_follow_up_questions(user_question, ai_response):
    """Generate contextual follow-up questions - DISABLED"""
    # Follow-up questions disabled to prevent interface issues
    return []

def generate_simple_followups(user_question, ai_response):
    """Simple fallback for follow-up generation when AI method fails"""
    combined_text = (user_question + " " + ai_response).lower()
    
    # Simple context-based follow-ups
    if 'dgft' in combined_text:
        return ["How to contact DGFT for this matter?", "What are the DGFT fees involved?", "How long does DGFT processing take?"]
    elif 'iec' in combined_text:
        return ["What documents are needed for this IEC process?", "How much does this IEC service cost?", "Can this be done online?"]
    elif 'export' in combined_text:
        return ["What are the export documentation requirements?", "How to handle export compliance?", "What export incentives are available?"]
    elif 'duty' in combined_text or 'drawback' in combined_text:
        return ["How is the duty/drawback calculated?", "What documents are required for claims?", "What are the time limits?"]
    else:
        return ["Can you explain this process step by step?", "What are the associated costs?", "How long does this typically take?"]

def display_suggested_questions():
    """Display suggested questions as clickable buttons"""
    st.markdown("### 💡 **Suggested Questions:**")
    
    suggestions = get_suggested_questions()
    
    # Display in a nice grid layout
    cols = st.columns(2)
    
    for i, suggestion in enumerate(suggestions[:8]):  # Show 8 suggestions
        col = cols[i % 2]
        with col:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # When clicked, simulate user input
                st.session_state.suggested_question = suggestion.split(' ', 1)[1] if ' ' in suggestion else suggestion
                st.rerun()

def display_follow_up_questions(questions, conversation_id):
    """Display follow-up questions - DISABLED for now"""
    # Follow-up questions disabled to prevent issues
    pass

def get_proactive_response_intro(user_question):
    """Generate a proactive intro to responses"""
    intros = [
        "Great question! Let me help you with that.",
        "I'd be happy to explain this for you.",
        "That's an important topic in international trade. Here's what you need to know:",
        "Excellent question! Based on the available information:",
        "Let me break this down for you:"
    ]
    
    # Choose intro based on question type
    if "?" in user_question:
        return intros[0]
    elif any(word in user_question.lower() for word in ['how', 'what', 'when', 'where', 'why']):
        return intros[1]
    elif any(word in user_question.lower() for word in ['help', 'assist', 'guide']):
        return intros[2]
    else:
        return intros[3]


def get_data_dir_hash():
    """Generate a hash based on files in data directory to detect changes."""
    import hashlib
    import os
    
    data_dir = "data/"
    if not os.path.exists(data_dir):
        return "no_data_dir"
    
    file_hash = hashlib.md5()
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(('.pdf', '.docx'))])
    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            # Include filename and modification time in hash
            file_hash.update(filename.encode('utf-8'))
            file_hash.update(str(os.path.getmtime(filepath)).encode('utf-8'))
    
    return file_hash.hexdigest()[:12]


@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def load_documents(data_dir_hash):
    """Load documents with caching that refreshes when files change."""
    try:
        # Ensure data directory exists
        data_dir = "data/"
        if not os.path.exists(data_dir):
            st.error(f"Data directory '{data_dir}' not found. Please ensure it exists on the server.")
            return []
        
        documents = load_documents_from_folder(data_dir)
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
    
    # Also clear TF-IDF cache files to force rebuild
    import glob
    cache_files = glob.glob("cache/*.pkl")
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            print(f"🗑️ Removed cache file: {cache_file}")
        except:
            pass
    
    # Force reload documents from folder with new hash
    data_dir_hash = get_data_dir_hash()
    return load_documents_from_folder("data/"), data_dir_hash


@st.cache_resource(show_spinner=False)
def initialize_components(documents, data_dir_hash):
    """Initialize chatbot components with caching that refreshes when documents change."""
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
        print(f"🚀 Initialized components for {session_id} (Data Hash: {data_dir_hash})")
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
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS and JS for better styling and image protection
    st.markdown('''
    <style>
    /* Remove top spacing and padding */
    .main .block-container {
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        margin-top: 0rem !important;
    }
    
    .stApp > div:first-child {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    .stApp {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    
    [data-testid="stAppViewContainer"] {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
    }
    
    /* Remove spacing from info elements */
    .stAlert {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }
    
    [data-testid="stAlert"] {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    
    /* Remove spacing from containers */
    .element-container {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }
    
    /* Hide Deploy button and menu dots */
    .stDeployButton {
        display: none !important;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    .stAppDeployButton {
        display: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    /* Hide the entire header toolbar */
    .stAppHeader {
        display: none !important;
    }
    
    header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Remove all default chat message styling first */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        margin: 0.2rem 0 !important;
    }
    
    /* Style user questions (odd numbered messages) with box and bold text */
    [data-testid="stChatMessage"]:nth-child(odd) {
        border: 1px solid #ddd !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        background: #f9f9f9 !important;
    }
    
    /* Make user question text bold */
    [data-testid="stChatMessage"]:nth-child(odd) p {
        font-weight: bold !important;
        color: #333 !important;
    }
    
    /* Ensure AI responses stay normal */
    [data-testid="stChatMessage"]:nth-child(even) {
        background: transparent !important;
        border: none !important;
        padding: 0.5rem 0 !important;
        margin: 0.2rem 0 !important;
    }
    
    [data-testid="stChatMessage"]:nth-child(even) p {
        font-weight: normal !important;
    }
    
    /* Hide empty chat messages completely */
    [data-testid="stChatMessage"]:empty {
        display: none !important;
    }
    
    /* Remove chat message avatars/icons */
    [data-testid="stChatMessage"] > div:first-child {
        display: none !important;
    }
    
    /* Style chat message content */
    [data-testid="stChatMessage"] > div {
        padding: 0 !important;
    }
    
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
    
    /* Custom chat input styling */
    .stChatInput input {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        border-radius: 25px !important;
        border: 2px solid #e0e0e0 !important;
        transition: border-color 0.3s ease !important;
    }
    
    .stChatInput input:focus {
        border-color: #4CAF50 !important;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2) !important;
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
    /* Set Inter font for entire app with 14px base size */
    .stApp, .stApp * {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
    }
    
    /* Chat message styling with Inter font */
    .stChatMessage {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    .stChatMessage p {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        margin-bottom: 0.8rem !important;
    }
    
    .stChatMessage ul, .stChatMessage ol {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    .stChatMessage li {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        margin-bottom: 0.6rem !important;
    }
    
    /* Chat input styling */
    .stChatInput > div > div > input {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        padding: 0.8rem !important;
    }
    
    /* All chat message containers */
    [data-testid="stChatMessage"] {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
        padding: 1rem !important;
        margin-bottom: 1rem !important;
    }
    
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] div {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }
    
    /* Headers and other text elements */
    h1, h2, h3, h4, h5, h6 {
        font-family: "Inter", sans-serif !important;
    }
    
    /* Buttons and UI elements */
    .stButton button {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
    }
    
    /* Sidebar elements */
    .sidebar .element-container {
        font-family: "Inter", sans-serif !important;
        font-size: 14px !important;
    }
    
    .chat-container {
        max-height: 500px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        background: #fafafa;
        font-family: "Inter", sans-serif !important;
    }
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-family: "Inter", sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Load documents initially and store in session state
    if 'documents' not in st.session_state or st.session_state.get('reload_documents', False):
        with st.spinner("Loading documents..."):
            # Generate hash to detect data directory changes
            data_dir_hash = get_data_dir_hash()
            st.session_state.documents = load_documents(data_dir_hash)
            st.session_state.data_dir_hash = data_dir_hash
            st.session_state.reload_documents = False
            # Clear the initialized flag to force reinitialization of components
            if st.session_state.get('reload_documents', False):
                st.session_state.pop('initialized', None)
                st.session_state.pop('retriever', None)
                st.session_state.pop('gemini', None)
    
    documents = st.session_state.documents
    data_dir_hash = st.session_state.get('data_dir_hash', 'unknown')
    
    # Check if data directory has changed since last load
    current_hash = get_data_dir_hash()
    if current_hash != data_dir_hash:
        st.info("🔄 New documents detected! Refreshing...")
        # Force reload with new hash
        st.session_state.documents = load_documents(current_hash)
        st.session_state.data_dir_hash = current_hash
        documents = st.session_state.documents
        data_dir_hash = current_hash
        # Clear initialization to force component refresh
        st.session_state.pop('initialized', None)
        st.session_state.pop('retriever', None)
        st.session_state.pop('gemini', None)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'show_suggestions' not in st.session_state:
        st.session_state.show_suggestions = True
    
    if 'conversation_context' not in st.session_state:
        st.session_state.conversation_context = []
    
    if 'last_followup_id' not in st.session_state:
        st.session_state.last_followup_id = None
    
    if 'shown_followups' not in st.session_state:
        st.session_state.shown_followups = set()  # Track shown questions to avoid repetition
    
    # Set default values for removed sidebar controls
    max_docs = 5  # Increase from 3 to 5 for better coverage
    show_sources = False  # Default to not showing sources
    
    if 'initialized' not in st.session_state:
        with st.spinner("Initializing AI components..."):
            retriever, gemini, error = initialize_components(documents, data_dir_hash)
            if error:
                st.error(f"❌ Error initializing AI: {error}")
                st.info("Please check your .env file and ensure GEMINI_API_KEY is set correctly.")
                return
            st.session_state.retriever = retriever
            st.session_state.gemini = gemini
            st.session_state.initialized = True
    
    # Main chat interface with improved layout
    
    # Add informational note about search capabilities
    st.markdown("""
    <div style="background-color: #e7f3ff; border: 1px solid #bee5eb; border-radius: 0.25rem; padding: 0.5rem; margin-bottom: 0.75rem; font-family: 'Inter', sans-serif !important; font-size: 12px;">
        <span style="font-family: 'Inter', sans-serif !important; font-size: 8ppx;">ⓘ <strong>Suggested Keywords:</strong> Try using terms like: DGFT, IEC, Export License, Import Policy, Duty Drawback, SEZ, Trade Documentation, Customs, EXIM Policy, Foreign Trade</span>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Simplified approach - no follow-up questions for now
    user_question = st.chat_input("Type your query...")
    
    # Display chat history with enhanced styling
    chat_container = st.container()
    with chat_container:
        if st.session_state.chat_history:
            for i, entry in enumerate(st.session_state.chat_history):
                format_chat_message("user", entry["question"], entry["timestamp"])
                
                if show_sources and entry.get("sources"):
                    with st.expander("📄 Sources used:"):
                        for source in entry["sources"]:
                            relevance = source.get('similarity_score', 0)
                            st.write(f"- **{source['file_name']}** (relevance: {relevance:.3f})")
                
                format_chat_message("assistant", entry["answer"], entry["timestamp"])
                st.divider()
    
    if user_question:
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Add user question to history immediately
        format_chat_message("user", user_question, timestamp)
        
        # Add proactive intro
        intro = get_proactive_response_intro(user_question)
        
        with st.spinner("🤖 Thinking... Analyzing your question and searching through documents..."):
            # Show thinking process
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🔍 Understanding your question...")
            progress_bar.progress(25)
            
            try:
                # For competency/capability questions, use more documents to get complete info
                search_terms = user_question.lower()
                if any(term in search_terms for term in ['competenc', 'skill', 'service', 'capabilit', 'core', 'all']):
                    # Use more documents for comprehensive answers
                    retrieval_count = min(max_docs + 2, 10)
                else:
                    retrieval_count = max_docs
                
                # Retrieve relevant documents
                status_text.text("📚 Searching through documents...")
                progress_bar.progress(50)
                
                relevant_docs = st.session_state.retriever.retrieve_relevant_chunks(
                    user_question, top_k=retrieval_count
                )
                
                # Special handling for ONLY EXPLICIT SVB queries - be very specific to avoid false matches
                if ('svb' in user_question.lower() and ('process' in user_question.lower() or 'flowchart' in user_question.lower() or 'flow chart' in user_question.lower())) or \
                   ('svb process flow chart' in user_question.lower()) or \
                   ('svb flowchart' in user_question.lower()) or \
                   ('svb document' in user_question.lower()) or \
                   ('special valuation branch' in user_question.lower()):
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
                
                # Generate answer with proactive intro
                status_text.text("🧠 Generating comprehensive answer...")
                progress_bar.progress(75)
                
                ai_response = st.session_state.gemini.chat_with_context(
                    user_question, 
                    relevant_docs, 
                    st.session_state.chat_history
                )
                
                # Combine intro with AI response
                enhanced_answer = f"{intro}\n\n{ai_response}"
                
                status_text.text("✅ Response ready!")
                progress_bar.progress(100)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Display answer
                format_chat_message("assistant", enhanced_answer, timestamp)
                
                # Save to session state
                st.session_state.chat_history.append({
                    "question": user_question,
                    "answer": enhanced_answer,
                    "sources": relevant_docs,
                    "timestamp": timestamp
                })
                
                # Rerun to update the display
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
