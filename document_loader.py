"""
Enhanced Document Loader with OCR Support
Handles PDF and DOCX files with image extraction and caching
"""

import os
import pickle
import hashlib
import json
from typing import List, Dict, Any
import PyPDF2
import fitz  # PyMuPDF
from docx import Document
import pytesseract
from PIL import Image
import io
import re


class DocumentLoader:
    """Enhanced document loader with OCR and caching capabilities"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Configure Tesseract path if needed (Windows)
        if os.name == 'nt':  # Windows
            try:
                # Try common Tesseract installation paths
                possible_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Users\admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        break
            except:
                pass  # Tesseract might be in PATH
    
    def ensure_cache_dir(self):
        """Create cache directory if it doesn't exist"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def save_to_cache(self, filepath: str, data: Dict[str, Any], file_type: str):
        """Save processed data to cache"""
        try:
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            cache_path = os.path.join(self.cache_dir, f"{base_name}_{file_type}_processed.json")
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Cached {file_type} results for {filename}")
        except Exception as e:
            print(f"⚠️  Failed to cache {file_type} results: {e}")
    
    def load_from_cache(self, filepath: str, file_type: str) -> Dict[str, Any]:
        """Load processed data from cache"""
        try:
            filename = os.path.basename(filepath)
            base_name = os.path.splitext(filename)[0]
            cache_path = os.path.join(self.cache_dir, f"{base_name}_{file_type}_processed.json")
            
            if os.path.exists(cache_path):
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"📁 Loading cached {file_type} results for {filename}")
                
                # Handle both old and new cache formats
                if 'document_data' in data:
                    # Old format
                    return data['document_data']
                else:
                    # New format
                    return data
            return None
        except Exception as e:
            print(f"⚠️  Failed to load cached {file_type} results: {e}")
            return None
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[str]:
        """Extract images from PDF and save to cache directory"""
        image_paths = []
        filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        try:
            doc = fitz.open(pdf_path)
            print(f"  🖼️  Extracting images from {len(doc)} pages...")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert to PNG if needed
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                        img_filename = f"{filename}_page{page_num + 1}_img{img_index}.png"
                        img_path = os.path.join(self.cache_dir, img_filename)
                        
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        image_paths.append(img_path)
                        print(f"    ✅ Extracted: {img_filename}")
                    
                    pix = None  # Free memory
            
            doc.close()
            print(f"  ✅ Total images extracted: {len(image_paths)}")
            
        except Exception as e:
            print(f"  ⚠️  Error extracting images: {e}")
        
        return image_paths
    
    def load_pdf_with_ocr(self, pdf_path: str) -> str:
        """Load PDF with OCR fallback and image extraction"""
        filename = os.path.basename(pdf_path)
        
        # Check cache first
        cached_data = self.load_from_cache(pdf_path, "pdf")
        if cached_data:
            return cached_data.get("content", "")
        
        print(f"Processing: {filename}")
        content = ""
        
        try:
            # Try regular text extraction first
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                print(f"PDF has {len(pdf_reader.pages)} pages")
                
                # Try to extract text from each page
                text_found = False
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        print(f"  Page {i+1}: Extracted {len(page_text)} characters")
                        content += page_text + "\n"
                        text_found = True
                    else:
                        print(f"  Page {i+1}: No text found")
                
                if text_found:
                    print(f"  ✅ Text-based PDF detected, extracting all pages...")
                    print(f"  ✅ Regular extraction: {len(content)} characters")
                else:
                    print(f"  ⚠️  No extractable text found, trying OCR...")
                    content = ""  # Reset content for OCR
        
        except Exception as e:
            print(f"  ⚠️  Error reading PDF: {e}")
        
        # Extract images (always do this for potential display)
        image_paths = self.extract_images_from_pdf(pdf_path)
        
        # If no text found, use OCR
        if not content.strip():
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    page_text = pytesseract.image_to_string(img, config='--psm 6')
                    if page_text.strip():
                        content += f"Page {page_num + 1}:\n{page_text}\n\n"
                
                doc.close()
                print(f"  ✅ OCR extraction: {len(content)} characters")
                
            except Exception as e:
                print(f"  ⚠️  OCR failed: {e}")
        
        # Add image references to content
        for i, img_path in enumerate(image_paths):
            rel_path = os.path.relpath(img_path, start=os.getcwd())
            content += f"\n[IMAGE_{i}: {rel_path}]\n"
        
        # Cache the results
        cache_data = {
            "content": content,
            "image_paths": image_paths,
            "processing_date": str(os.path.getmtime(pdf_path))
        }
        self.save_to_cache(pdf_path, cache_data, "pdf")
        
        return content
    
    def load_docx(self, docx_path: str) -> str:
        """Load DOCX file with image extraction"""
        filename = os.path.basename(docx_path)
        
        # Check cache first
        cached_data = self.load_from_cache(docx_path, "docx")
        if cached_data:
            return cached_data.get("content", "")
        
        print(f"Processing: {filename}")
        content = ""
        image_paths = []
        
        try:
            doc = Document(docx_path)
            
            # Extract text
            for paragraph in doc.paragraphs:
                content += paragraph.text + "\n"
            
            # Extract images from DOCX
            for rel in doc.part.rels.values():
                if "image" in rel.target_ref:
                    try:
                        img_data = rel.target_part.blob
                        img_filename = f"{os.path.splitext(filename)[0]}_{len(image_paths)}.png"
                        img_path = os.path.join(self.cache_dir, img_filename)
                        
                        # Save image
                        with open(img_path, "wb") as img_file:
                            img_file.write(img_data)
                        
                        image_paths.append(img_path)
                        
                        # Add image reference to content
                        rel_path = os.path.relpath(img_path, start=os.getcwd())
                        content += f"\n[IMAGE_{len(image_paths)-1}: {rel_path}]\n"
                        
                    except Exception as e:
                        print(f"  ⚠️  Error extracting image: {e}")
            
            print(f"Loaded: {filename} ({len(content)} characters)")
            
            # Cache the results
            cache_data = {
                "content": content,
                "image_paths": image_paths,
                "processing_date": str(os.path.getmtime(docx_path))
            }
            self.save_to_cache(docx_path, cache_data, "docx")
            
        except Exception as e:
            print(f"  ⚠️  Error loading DOCX: {e}")
        
        return content
    
    def load_document(self, filepath: str) -> str:
        """Load a single document based on its type"""
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext == '.pdf':
            return self.load_pdf_with_ocr(filepath)
        elif ext == '.docx':
            return self.load_docx(filepath)
        else:
            print(f"  ⚠️  Unsupported file type: {ext}")
            return ""


def load_documents_from_folder(folder_path: str) -> List[Dict[str, Any]]:
    """
    Load all documents from a folder and return a list of document objects
    """
    print(f"Loading documents from: {os.path.abspath(folder_path)}")
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    loader = DocumentLoader()
    documents = []
    
    # Get all supported files
    supported_extensions = ['.pdf', '.docx']
    files = []
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in supported_extensions:
                files.append(filepath)
    
    if not files:
        print("❌ No supported documents found (PDF, DOCX)")
        return []
    
    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")
    
    # Process each file
    for filepath in files:
        try:
            content = loader.load_document(filepath)
            
            if content.strip():
                documents.append({
                    'file_name': os.path.basename(filepath),  # Changed from 'filename' to 'file_name'
                    'filepath': filepath,
                    'content': content,
                    'type': os.path.splitext(filepath)[1].lower()
                })
            else:
                print(f"⚠️  {os.path.basename(filepath)}: No content extracted")
        except Exception as e:
            print(f"❌ Failed to load {os.path.basename(filepath)}: {e}")
    
    print(f"Successfully loaded {len(documents)} documents.")
    return documents


if __name__ == "__main__":
    """Test the document loader"""
    documents = load_documents_from_folder("data/")
    for doc in documents:
        print(f"📄 {doc['file_name']}: {len(doc['content'])} characters")
