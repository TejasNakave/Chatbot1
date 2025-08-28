import os
import glob
import json
import hashlib
from typing import List, Dict
import PyPDF2
from docx import Document

# OCR imports with fallback
try:
    import fitz  # PyMuPDF
    import pytesseract
    from PIL import Image
    import io
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def get_file_hash(file_path: str) -> str:
    """Get MD5 hash of file for caching purposes"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_cache_path(file_path: str) -> str:
    """Get cache file path for OCR results"""
    cache_dir = "cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    file_name = os.path.basename(file_path)
    cache_name = f"{os.path.splitext(file_name)[0]}_ocr_cache.json"
    return os.path.join(cache_dir, cache_name)


def load_from_cache(file_path: str) -> str:
    """Load OCR results from cache if available and file hasn't changed"""
    cache_path = get_cache_path(file_path)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if file hash matches (file hasn't changed)
        current_hash = get_file_hash(file_path)
        if cache_data.get('file_hash') == current_hash:
            print(f"📁 Loading cached OCR results for {os.path.basename(file_path)}")
            return cache_data.get('content')
        else:
            print(f"🔄 File changed, cache invalid for {os.path.basename(file_path)}")
            return None
            
    except Exception as e:
        print(f"⚠️  Cache read error: {str(e)}")
        return None


def save_to_cache(file_path: str, content: str):
    """Save OCR results to cache"""
    cache_path = get_cache_path(file_path)
    
    try:
        cache_data = {
            'file_hash': get_file_hash(file_path),
            'content': content,
            'cached_at': str(os.path.getmtime(file_path))
        }
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved OCR results to cache for {os.path.basename(file_path)}")
        
    except Exception as e:
        print(f"⚠️  Cache save error: {str(e)}")


def load_documents_from_folder(folder_path: str = "data/") -> List[Dict[str, str]]:
    """
    Load all .pdf, .docx, and .txt files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing documents
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing file content and metadata
    """
    documents = []
    
    # Ensure folder exists
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' does not exist.")
        return documents
    
    print(f"Loading documents from: {os.path.abspath(folder_path)}")
    
    # Get all supported file types
    file_patterns = [
        os.path.join(folder_path, "*.pdf"),
        os.path.join(folder_path, "*.docx"),
        os.path.join(folder_path, "*.txt")
    ]
    
    all_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        all_files.extend(files)
    
    print(f"Found {len(all_files)} files: {[os.path.basename(f) for f in all_files]}")
    
    for file_path in all_files:
        try:
            content = ""
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            print(f"Processing: {file_name}")
            
            if file_ext == ".pdf":
                content = load_pdf(file_path)
            elif file_ext == ".docx":
                content = load_docx(file_path)
            elif file_ext == ".txt":
                content = load_txt(file_path)
            
            if content.strip():  # Only add if content is not empty
                documents.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "content": content,
                    "file_type": file_ext
                })
                print(f"Loaded: {file_name} ({len(content)} characters)")
            else:
                print(f"Warning: No content found in {file_name}")
                # Still add the document with a note that it couldn't be processed
                documents.append({
                    "file_name": file_name,
                    "file_path": file_path,
                    "content": f"This file could not be processed. File: {file_name}\nReason: No extractable text content found.",
                    "file_type": file_ext
                })
                print(f"Added with placeholder content: {file_name}")
                
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Successfully loaded {len(documents)} documents.")
    return documents


def extract_text_with_ocr(pdf_path: str, max_pages: int = 215) -> str:
    """Extract text from scanned PDF using OCR with caching"""
    if not OCR_AVAILABLE:
        return f"OCR not available for {os.path.basename(pdf_path)}"
    
    # Check cache first
    cached_content = load_from_cache(pdf_path)
    if cached_content:
        return cached_content
    
    # Set up Tesseract path
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
    try:
        print(f"🔍 Running OCR on {os.path.basename(pdf_path)} ({max_pages} pages)...")
        
        # Open PDF with PyMuPDF
        doc = fitz.open(pdf_path)
        extracted_text = ""
        successful_pages = 0
        
        total_pages = min(len(doc), max_pages)
        print(f"📄 Processing {total_pages} pages with OCR...")
        
        for page_num in range(total_pages):
            try:
                if page_num % 10 == 0:  # Progress update every 10 pages
                    print(f"   � Progress: {page_num + 1}/{total_pages} pages...")
                
                # Get page and convert to image
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                # Run OCR
                text = pytesseract.image_to_string(img, lang='eng')
                
                if text.strip():
                    extracted_text += f"\n=== Page {page_num + 1} ===\n{text.strip()}\n"
                    successful_pages += 1
                    
            except Exception as e:
                print(f"   ❌ Page {page_num + 1}: Error - {str(e)}")
        
        doc.close()
        
        if extracted_text.strip():
            full_text = f"""OCR-Extracted Text from {os.path.basename(pdf_path)}
Source: Scanned PDF processed with OCR
Pages processed: {successful_pages}/{total_pages} (of {len(fitz.open(pdf_path))} total)

{extracted_text}

Note: Full document processed and cached for faster future access."""
            
            print(f"✅ OCR completed: {len(full_text)} characters from {successful_pages} pages")
            
            # Save to cache for faster future loading
            save_to_cache(pdf_path, full_text)
            
            return full_text
        else:
            return f"OCR found no readable text in {os.path.basename(pdf_path)}"
            
    except Exception as e:
        print(f"❌ OCR failed: {str(e)}")
        return f"OCR processing failed for {os.path.basename(pdf_path)}: {str(e)}"


def load_pdf(file_path: str) -> str:
    """Extract text from PDF file with improved handling, OCR fallback, and image extraction."""
    text = ""
    images = []
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f"PDF has {len(pdf_reader.pages)} pages")
            
            # Check first few pages for regular text
            pages_with_text = 0
            for page_num in range(min(5, len(pdf_reader.pages))):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
                        pages_with_text += 1
                        print(f"  Page {page_num + 1}: Extracted {len(page_text)} characters")
                    else:
                        print(f"  Page {page_num + 1}: No text found")
                except Exception as e:
                    print(f"  Page {page_num + 1}: Error extracting text - {str(e)}")
            
            # If we found text, continue with regular extraction
            if pages_with_text > 0:
                print(f"  ✅ Text-based PDF detected, extracting all pages...")
                for page in pdf_reader.pages[5:]:
                    try:
                        text += page.extract_text() + "\n"
                    except:
                        pass
                print(f"  ✅ Regular extraction: {len(text)} characters")
                
                # Extract images using PyMuPDF (even for text-based PDFs)
                images = extract_images_from_pdf(file_path)
            else:
                # No regular text found, try OCR
                print(f"  ⚠️  No extractable text found, trying OCR...")
                text = extract_text_with_ocr(file_path, max_pages=215)
                
                # Extract images for OCR-based PDFs too
                images = extract_images_from_pdf(file_path)
        
        # Add image references to text if images were found
        if images:
            text += f"\n[IMAGES_AVAILABLE: {len(images)} images extracted]\n"
            for i, img in enumerate(images):
                text += f"[IMAGE_{i}: {img['path']}]\n"
                
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")
    return text


def extract_images_from_pdf(file_path: str) -> list:
    """Extract images from PDF using PyMuPDF."""
    images = []
    
    if not OCR_AVAILABLE:  # PyMuPDF is imported with OCR
        return images
    
    try:
        doc = fitz.open(file_path)
        file_base = os.path.splitext(os.path.basename(file_path))[0]
        
        # Create cache directory
        cache_dir = "cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        print(f"  🖼️  Extracting images from {len(doc)} pages...")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    # Convert CMYK to RGB if needed
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_data = pix.tobytes("png")
                    else:  # CMYK
                        pix1 = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix1.tobytes("png")
                        pix1 = None
                    
                    # Save image
                    image_name = f"{file_base}_page{page_num+1}_img{img_index}.png"
                    image_path = os.path.join(cache_dir, image_name)
                    
                    # Use forward slashes for cross-platform compatibility
                    relative_path = image_path.replace('\\', '/')
                    
                    with open(image_path, 'wb') as img_file:
                        img_file.write(img_data)
                    
                    images.append({
                        'path': relative_path,
                        'name': image_name,
                        'page': page_num + 1
                    })
                    
                    print(f"    ✅ Extracted: {image_name}")
                    
                    pix = None
                    
                except Exception as e:
                    print(f"    ❌ Error extracting image {img_index} from page {page_num + 1}: {e}")
        
        doc.close()
        print(f"  ✅ Total images extracted: {len(images)}")
        
    except Exception as e:
        print(f"  ❌ Error extracting images from PDF: {e}")
    
    return images


def load_docx(file_path: str) -> str:
    """Extract text from DOCX file with image extraction support."""
    try:
        doc = Document(file_path)
        text = ""
        images = []
        
        # Extract images from the document
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                try:
                    image_data = rel.target_part.blob
                    image_name = f"{os.path.splitext(os.path.basename(file_path))[0]}_{len(images)}.png"
                    
                    # Save image to cache directory
                    cache_dir = "cache"
                    if not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    
                    image_path = os.path.join(cache_dir, image_name)
                    
                    with open(image_path, 'wb') as img_file:
                        img_file.write(image_data)
                    
                    # Use forward slashes for cross-platform compatibility in Streamlit
                    relative_path = image_path.replace('\\', '/')
                    images.append({
                        'path': relative_path,  # Use forward slashes like "cache/image.png"
                        'name': image_name
                    })
                except Exception as e:
                    print(f"Warning: Could not extract image: {str(e)}")
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # If images were found, add reference to them in the text
        if images:
            text += f"\n[IMAGES_AVAILABLE: {len(images)} images extracted]\n"
            for i, img in enumerate(images):
                text += f"[IMAGE_{i}: {img['path']}]\n"
                
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")
    return text


def load_txt(file_path: str) -> str:
    """Load text from TXT file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
        except Exception as e:
            raise Exception(f"Error reading TXT file with multiple encodings: {str(e)}")
    except Exception as e:
        raise Exception(f"Error reading TXT: {str(e)}")
    return text


if __name__ == "__main__":
    # Test the document loader
    docs = load_documents_from_folder()
    for doc in docs:
        print(f"\nFile: {doc['file_name']}")
        print(f"Type: {doc['file_type']}")
        print(f"Content preview: {doc['content'][:200]}...")
