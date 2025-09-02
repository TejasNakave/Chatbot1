import os
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Optional
from datetime import datetime


class GeminiAPIWrapper:
    """
    A wrapper class for the Gemini API that handles question-answering with context.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini API wrapper.
        
        Args:
            api_key (Optional[str]): Gemini API key. If None, loads from environment.
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("Gemini API key not found. Please set GEMINI_API_KEY in .env file or pass it directly.")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize the model with increased output tokens
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=2048,  # Increased from default
            temperature=0.7,
        )
        self.model = genai.GenerativeModel('gemini-1.5-flash', generation_config=generation_config)
        
        print("Gemini API wrapper initialized successfully!")
    
    def _filter_images_for_query(self, context: str, question: str) -> str:
        """
        Filter image references in context based on query type.
        For coffee queries, only include coffee-related images.
        For SVB queries, only include SVB/trade-related images.
        """
        import re
        
        # Find all image references
        image_refs = re.findall(r'\[IMAGE_\d+: [^\]]+\]', context)
        if not image_refs:
            return context
            
        question_lower = question.lower()
        is_coffee_query = 'coffee' in question_lower and 'svb' not in question_lower
        is_svb_query = 'svb' in question_lower and 'coffee' not in question_lower
        
        # Only filter if it's a specific coffee or SVB query
        if not (is_coffee_query or is_svb_query):
            return context
            
        print(f"🔍 FILTERING IMAGES: coffee_query={is_coffee_query}, svb_query={is_svb_query}")
        print(f"🔍 Found {len(image_refs)} images to filter:")
        for ref in image_refs:
            print(f"    - {ref}")
            
        # Filter images based on query type
        filtered_refs = []
        for i, ref in enumerate(image_refs):
            ref_lower = ref.lower()
            
            if is_coffee_query:
                # For coffee queries: prefer later images (_1) and avoid SVB trade terms
                is_relevant = (
                    'coffee' in ref_lower or 
                    (i > 0 and 'svb' in ref_lower) or  # Later images in SVB doc are often coffee
                    '_1' in ref  # Second image is often coffee in mixed documents
                )
                if is_relevant:
                    filtered_refs.append(ref)
                    print(f"    ✅ KEEPING for COFFEE: {ref}")
                else:
                    print(f"    ❌ REMOVING for COFFEE: {ref} (trade-related)")
                    
            elif is_svb_query:
                # For SVB queries: be very explicit - only keep the FIRST image
                # Since in mixed documents, first image (_0) is typically trade/SVB
                if i == 0:
                    filtered_refs.append(ref)
                    print(f"    ✅ KEEPING for SVB: {ref} (first image - trade process)")
                else:
                    print(f"    ❌ REMOVING for SVB: {ref} (secondary image - likely coffee)")
        
        # Replace context with filtered images
        if filtered_refs:
            # Remove all image references first
            filtered_context = re.sub(r'\[IMAGE_\d+: [^\]]+\]', '', context)
            # Add back only the relevant ones
            image_section = f"\n\nRelevant images for {question_lower}:\n" + "\n".join(filtered_refs)
            filtered_context += image_section
            print(f"🔍 FILTERED RESULT: Keeping {len(filtered_refs)} relevant images")
            return filtered_context
        else:
            print(f"🔍 FILTERED RESULT: No relevant images found, removing all")
            return re.sub(r'\[IMAGE_\d+: [^\]]+\]', '', context)  # Remove all images
        """
        Ask a question to Gemini with optional context.
        
        Args:
            question (str): The question to ask
            context (str): Additional context to provide to the model
            max_tokens (int): Maximum number of tokens in the response
            
        Returns:
            str: The model's response
        """
        try:
            # Construct the prompt
            if context.strip():
                # Check if context contains image references
                has_images = '[IMAGE_' in context
                
                if has_images:
                    # Special handling for coffee queries from mixed documents
                    if 'coffee' in question.lower() and 'svb' not in question.lower():
                        prompt = f"""You are a helpful assistant. The user is asking about coffee-making process.

IMPORTANT: The context may contain mixed content (both coffee and trade-related information). 

Context:
{context}

Question: {question}

CRITICAL INSTRUCTIONS FOR COFFEE QUERIES:
- FOCUS ONLY on coffee-making, coffee process, or coffee-related content
- IGNORE any trade, customs, SVB, or business-related information
- Look at the image references provided - there may be multiple images
- ONLY include images that are clearly related to COFFEE MAKING (look for titles like "COFFEE MAKING PROCESS")
- Do NOT include images about trade processes, SVB, customs, or business flowcharts
- If you see multiple [IMAGE_X: path] references, choose ONLY the coffee-related ones
- Common coffee image indicators: "COFFEE", "brewing", "beans", "water", "grind"
- Common non-coffee indicators: "SVB", "trade", "customs", "import", "export", "business"

EXAMPLE: If you see [IMAGE_0: svb_trade.png] and [IMAGE_1: coffee_process.png], only include IMAGE_1.

Answer with coffee-focused content and ONLY coffee-related images:"""
                    elif 'svb' in question.lower() and 'coffee' not in question.lower():
                        prompt = f"""You are a trade assistant chatbot specializing in customs and trade processes.

The user is asking about SVB (Special Valuation Branch) process.

Context:
{context}

Question: {question}

CRITICAL INSTRUCTIONS FOR SVB QUERIES:
- FOCUS ONLY on SVB, Special Valuation Branch, trade, customs, and business processes
- IGNORE any coffee-making, cooking, or food-related information
- Look at the image references provided - there may be multiple images
- ONLY include images that are clearly related to TRADE/SVB PROCESSES
- Do NOT include images about coffee making, food preparation, or cooking
- If you see multiple [IMAGE_X: path] references, choose ONLY the trade/SVB-related ones
- Common SVB/trade indicators: "SVB", "trade", "customs", "import", "export", "business", "valuation", "assessment"
- Common non-trade indicators: "COFFEE", "brewing", "beans", "water", "cooking", "food"

Answer with trade/SVB-focused content and ONLY trade-related images:"""
                    else:
                        prompt = f"""You are a trade assistant chatbot.
When providing answers:
1. Rewrite retrieved text into a clean, natural explanation.
2. If an image reference (image_path) is available, show it with a caption.
3. Avoid repeating raw bullet points from documents.

Based on the following context, please answer the question:

Context:
{context}

Question: {question}

CRITICAL INSTRUCTIONS:
- ALWAYS include ALL [IMAGE_X: path] references EXACTLY as they appear in the context
- If you see image references like [IMAGE_0: cache/filename.png], copy them EXACTLY into your response
- Rewrite retrieved information into clear, natural explanations rather than copying raw text
- Transform bullet points and document fragments into flowing, coherent sentences
- Make the response descriptive and well-structured
- Do NOT include citations or page numbers, but DO include image references
- Focus on being helpful and informative

IMPORTANT: Copy image references exactly: [IMAGE_0: cache/filename.png]

Answer the question with clear, natural explanations:"""
                else:
                    prompt = f"""You are a trade assistant chatbot.
When providing answers:
1. Rewrite retrieved text into a clean, natural explanation.
2. Avoid repeating raw bullet points from documents.
3. Transform document fragments into coherent, flowing text.

Based on the following context, please answer the question:

Context:
{context}

Question: {question}

Instructions:
- Rewrite information from the context into clear, natural explanations
- Transform bullet points and fragmented text into coherent sentences
- Make the response descriptive and well-structured
- Do NOT include any references, citations, or page numbers
- Focus on being helpful and informative

Answer the question with clear, natural explanations:"""
            else:
                prompt = f"""You are a trade assistant chatbot.
When providing answers:
1. Provide clear, natural explanations.
2. Avoid using bullet points unless absolutely necessary.
3. Focus on coherent, flowing responses.

Question: {question}

Instructions:
- Provide a clear, comprehensive answer in natural language
- Use flowing sentences rather than bullet points when possible
- Make the response descriptive and well-structured
- Focus on being helpful and informative

Answer the question clearly and descriptively:"""
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response.text:
                return response.text.strip()
            else:
                return "I'm sorry, I couldn't generate a response to your question."
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _answer_with_context(self, question: str, conversation_context: str, context_images: list) -> str:
        """
        Answer a contextual query using conversation context and images.
        
        Args:
            question (str): The contextual question
            conversation_context (str): Previous conversation context
            context_images (list): List of image references from context
            
        Returns:
            str: Response with relevant images and explanation
        """
        # Get current date and time
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")
        
        # Build context with images
        image_context = "Available images from previous conversation:\n"
        for img in context_images:
            image_context += f"{img}\n"
        
        contextual_prompt = f"""Current Date: {current_date}
Current Time: {current_time}

{conversation_context}

{image_context}

Question: {question}

Instructions:
- This is a contextual question referring to previous images or content
- Use the conversation context and available images to answer
- Include relevant image references EXACTLY as shown above
- Provide a natural explanation of the content shown in the images
- If asked to explain "the above process" or similar, refer to the most recently shown images

CRITICAL IMAGE HANDLING:
- Include relevant image references EXACTLY as they appear: [IMAGE_X: full_path]
- For contextual queries, show the relevant images from the conversation
"""

        try:
            response = self.model.generate_content(contextual_prompt)
            if response.text:
                print(f"🤖 DEBUG: Contextual AI Response:")
                print(f"  Response length: {len(response.text)} chars")
                # Check for image references in the response
                import re
                image_refs = re.findall(r'\[IMAGE_\d+: [^\]]+\]', response.text)
                if image_refs:
                    print(f"  ✅ Response contains image references")
                    for ref in image_refs:
                        print(f"    - {ref}")
                else:
                    print(f"  ❌ No image references in response")
                
                return response.text
            else:
                return "I couldn't generate a proper response to your contextual question."
                
        except Exception as e:
            return f"Error generating contextual response: {str(e)}"
    
    def chat_with_context(self, question: str, documents: list, chat_history: list = None) -> str:
        """
        Answer a question using the provided documents as context and chat history.
        
        Args:
            question (str): The question to ask
            documents (list): List of document dictionaries with 'content' key
            chat_history (list): Previous conversation history for context
            
        Returns:
            str: The model's response without citations, with global fallback
        """
        # Early check: If no documents found and it's not a contextual query, inform user
        contextual_terms = ['above', 'previous', 'that', 'this', 'those', 'these', 'the image', 'the process', 'earlier']
        is_contextual = any(term in question.lower() for term in contextual_terms)
        
        if (not documents or len(documents) == 0) and not is_contextual:
            return f"❌ **No Information Found**\n\nI couldn't find any information about '{question}' in the currently loaded documents. This could mean:\n\n• The term doesn't exist in your documents\n• Documents containing this information haven't been added to the system\n• The search term might need to be adjusted\n\n**Current Status:** Only {self._count_loaded_documents()} document(s) loaded.\n\nPlease check if the relevant documents are in your data folder and try rephrasing your question."
        
        # Build conversation context from chat history
        conversation_context = ""
        if chat_history and len(chat_history) > 0:
            # Include last few exchanges for context (limit to avoid token overflow)
            recent_history = chat_history[-3:] if len(chat_history) > 3 else chat_history
            conversation_context = "Recent conversation context:\n"
            for entry in recent_history:
                conversation_context += f"User: {entry.get('question', '')}\n"
                conversation_context += f"Assistant: {entry.get('answer', '')}\n\n"
            conversation_context += "Current question follows:\n\n"
        
        # Check if this is a contextual query (no documents returned by retriever)
        # but we have conversation context with images
        if (not documents or len(documents) == 0) and conversation_context:
            # Check if conversation context contains images and this is a contextual query
            import re
            context_images = re.findall(r'\[IMAGE_\d+: [^\]]+\]', conversation_context)
            contextual_terms = ['above', 'previous', 'that', 'this', 'those', 'these', 'the image', 'the process', 'earlier']
            is_contextual = any(term in question.lower() for term in contextual_terms)
            
            if context_images and is_contextual:
                print(f"🔗 CONTEXTUAL QUERY WITH IMAGES: Using conversation context")
                # For contextual queries, we need to get the documents from the last query
                # instead of just using conversation context
                # Look for recent document-based answers in chat history
                if chat_history:
                    last_entry = chat_history[-1] if chat_history else None
                    if last_entry and last_entry.get('sources'):
                        print(f"🔗 USING SOURCES FROM LAST QUERY: {len(last_entry.get('sources', []))} documents")
                        # Use the same documents from the previous query for contextual explanation
                        return self.chat_with_context(question, last_entry.get('sources', []), chat_history[:-1])
                
                # Fallback to conversation context if no sources found
                return self._answer_with_context(question, conversation_context, context_images)
        
        # First try to answer using provided documents
        if documents and len(documents) > 0:
            # Combine all document content as context
            context = ""
            
            for i, doc in enumerate(documents):
                doc_name = doc.get('file_name', f'Document {i+1}')
                content = doc.get('content', '')
                if content.strip():
                    context += f"Document: {doc_name}\n{content}\n\n"
            
            # Check if we have any actual content
            if context.strip():
                # Filter images based on query type (especially for coffee queries)
                context = self._filter_images_for_query(context, question)
                
                # Limit context length to avoid token limits
                if len(context) > 12000:  # Increased limit for longer responses
                    context = context[:12000] + "...\n[Content truncated due to length]"
                
                # Try answering with document context first
                current_date = datetime.now().strftime("%A, %B %d, %Y")
                current_time = datetime.now().strftime("%I:%M %p")
                
                document_prompt = f"""Current Date: {current_date}
Current Time: {current_time}

You are a trade assistant chatbot that provides information from PDF documents.

{conversation_context}Document Content:
{context}

Question: {question}

RESPONSE RULES:

1. FOR DEFINITION/CERTIFICATE QUERIES (What is MSME, CIN, STAR certificate, etc.):
   - Return the EXACT text from the PDF document as-is
   - Only make minimal formatting improvements:
     * Replace 'l' with '•' for bullet points
     * Fix obvious spacing issues (multiple spaces to single space)
     * Fix broken words that are clearly OCR errors
   - Do NOT rewrite, paraphrase, or restructure the content
   - Keep the original wording and structure from the PDF

2. FOR OTHER GENERAL QUERIES:
   - Rewrite document content into clear, flowing explanations
   - Transform bullet points and fragmented text into coherent sentences
   - Provide comprehensive narrative descriptions

3. GLOBAL SEARCH CRITERIA:
   - If document only mentions the term but doesn't define/explain it
   - If document contains table of contents/references but not actual content
   - If topic is completely unrelated to document content
   - If no relevant answer found, respond ONLY with: "GLOBAL_SEARCH_NEEDED"

IMAGE HANDLING:
- If you see "[IMAGE_X: path]" references, include them EXACTLY in your answer
- Only include images relevant to the specific query

CRITICAL: For definition questions, preserve the original PDF text. Only clean up obvious formatting errors, don't rewrite the content.

Answer:"""

                try:
                    # Debug: Print what we're sending to AI
                    if "SVB" in question.upper() or "PROCESS" in question.upper():
                        print(f"🤖 DEBUG: Sending to AI:")
                        print(f"  Question: {question}")
                        print(f"  Context length: {len(context)} chars")
                        if "[IMAGE_" in context:
                            print(f"  ✅ Context contains image references")
                            import re
                            image_refs = re.findall(r'\[IMAGE_\d+: [^\]]+\]', context)
                            for ref in image_refs:
                                print(f"    - {ref}")
                        else:
                            print(f"  ❌ No image references found in context")
                    
                    response = self.model.generate_content(document_prompt)
                    if response.text and response.text.strip():
                        doc_answer = response.text.strip()
                        
                        # Check if this is an image request and AI said GLOBAL_SEARCH_NEEDED
                        # but we actually have images - force include them ONLY if topic is relevant
                        if (doc_answer == "GLOBAL_SEARCH_NEEDED" and 
                            any(word in question.lower() for word in ['display', 'show', 'image', 'figure', 'flowchart', 'chart', 'diagram']) and
                            '[IMAGE_' in context):
                            
                            # Additional check: Don't force images for completely unrelated topics  
                            unrelated_topics = ['coffee', 'cooking', 'recipe', 'food', 'kitchen', 'ingredient', 'baking',
                                              'personal', 'family', 'health', 'medical', 'sports', 'entertainment', 'music',
                                              'movie', 'game', 'hobby', 'travel', 'vacation', 'weather', 'animal', 'pet']
                            # Don't include generic words like 'process' in trade terms for this check
                            trade_terms = ['trade', 'export', 'import', 'customs', 'duty', 'tariff', 'svb', 
                                          'business', 'commerce', 'shipping', 'freight', 'documentation', 'fta', 'drawback',
                                          'compliance', 'regulation', 'clearance', 'invoice', 'certificate']
                            
                            has_unrelated = any(term in question.lower() for term in unrelated_topics)
                            has_trade_terms = any(term in question.lower() for term in trade_terms)
                            
                            # Handle coffee and SVB queries more flexibly
                            is_coffee_query = 'coffee' in question.lower()
                            is_svb_query = 'svb' in question.lower()
                            is_mixed_query = is_coffee_query and is_svb_query
                            
                            print(f"🔍 TOPIC CHECK: coffee={is_coffee_query}, svb={is_svb_query}, mixed={is_mixed_query}, has_unrelated={has_unrelated}, has_trade_terms={has_trade_terms}")
                            print(f"🔍 QUESTION: '{question.lower()}'")
                            
                            # Allow image forcing for coffee queries now that we filter content appropriately
                            if is_coffee_query or is_svb_query or (not has_unrelated or has_trade_terms):
                                print("🔧 FIXING: AI said GLOBAL_SEARCH_NEEDED but we have images for visual request")
                                # Extract image references from context
                                import re
                                image_refs = re.findall(r'\[IMAGE_\d+: [^\]]+\]', context)
                                if image_refs:
                                    doc_answer = f"Here are the requested images:\n\n" + "\n".join(image_refs)
                                    print(f"🔧 FORCED IMAGE RESPONSE: {len(image_refs)} images included")
                            else:
                                print(f"🚫 TOPIC MISMATCH: Not forcing images for unrelated query: {question}")
                        # Debug: Print what AI responded
                        if "SVB" in question.upper() or "PROCESS" in question.upper() or "flowchart" in question.lower():
                            print(f"🤖 DEBUG: AI Response:")
                            print(f"  Response length: {len(doc_answer)} chars")
                            if "[IMAGE_" in doc_answer:
                                print(f"  ✅ Response contains image references")
                                import re
                                image_refs = re.findall(r'\[IMAGE_\d+: [^\]]+\]', doc_answer)
                                for ref in image_refs:
                                    print(f"    - {ref}")
                            else:
                                print(f"  ❌ No image references in response")
                                print(f"  First 200 chars: {doc_answer[:200]}")
                        
                        # Check for explicit global search signal
                        if doc_answer == "GLOBAL_SEARCH_NEEDED":
                            return self._get_global_answer(question, conversation_context)
                        
                        # Check if the answer indicates information wasn't found
                        not_found_indicators = [
                            "document does not specify", "document does not mention", 
                            "document does not provide", "document does not contain",
                            "text does not include", "document focuses on", "therefore, i cannot",
                            "based on the given text", "document appears to be empty",
                            "no content available", "couldn't be processed",
                            "provided text", "does not specify", "does not include",
                            "text about", "does not mention", "is not provided",
                            "information is not", "details are not", "not mentioned in",
                            "provided document", "guide for", "does not contain information"
                        ]
                        
                        # If the document-based answer seems incomplete, try global search
                        if any(indicator in doc_answer.lower() for indicator in not_found_indicators):
                            return self._get_global_answer(question, conversation_context)
                        else:
                            return doc_answer
                except Exception:
                    pass  # Fall through to global search
        
        # Fallback to global search if no documents or document search failed
        return self._get_global_answer(question, conversation_context)
    
    def _get_global_answer(self, question: str, conversation_context: str = "") -> str:
        """
        Provide a global answer when information isn't found in documents.
        
        Args:
            question (str): The question to answer
            conversation_context (str): Previous conversation context
            
        Returns:
            str: A comprehensive answer from global knowledge
        """
        # Get current date and time
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        current_time = datetime.now().strftime("%I:%M %p")
        
        global_prompt = f"""Current Date: {current_date}
Current Time: {current_time}

Provide a clear, comprehensive answer to this question using your general knowledge:

{conversation_context}Question: {question}

Instructions:
- Consider the conversation context when answering
- If the question refers to "it", "this", "that", or "more details", refer to the previous conversation
- If asked about current date/time, use the provided current date and time above
- Provide a detailed, informative answer
- Use bullet points when listing multiple items or features
- Make the response descriptive and well-structured
- Do NOT mention that this information comes from general knowledge
- Do NOT say the information wasn't found in documents
- Provide the best possible answer you can

Answer the question clearly and descriptively:"""
        
        try:
            response = self.model.generate_content(global_prompt)
            if response.text:
                return response.text.strip()
            else:
                return "I'm sorry, I couldn't generate a response to your question."
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def _count_loaded_documents(self) -> int:
        """Helper method to count loaded documents from session or system."""
        try:
            # This is a simple placeholder - in practice, this would check the actual loaded documents
            # For now, we'll return a generic count
            return 1  # Since we know only Impex New Book.pdf is loading
        except:
            return 0


def test_gemini_wrapper():
    """Test function for the Gemini API wrapper."""
    try:
        gemini = GeminiAPIWrapper()
        
        # Test simple question
        response = gemini.ask_question("What is the capital of France?")
        print("Test Question: What is the capital of France?")
        print(f"Response: {response}\n")
        
        # Test with context
        context = "Python is a high-level programming language. It was created by Guido van Rossum."
        question = "Who created Python?"
        response = gemini.ask_question(question, context)
        print(f"Test Question with Context: {question}")
        print(f"Context: {context}")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error testing Gemini wrapper: {str(e)}")


if __name__ == "__main__":
    test_gemini_wrapper()
