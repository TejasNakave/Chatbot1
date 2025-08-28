# 🤖 Document-Based AI Chatbot

> **Powerful AI chatbot with OCR support** - Chat with your documents using Google Gemini API  
> Supports PDF, DOCX, TXT files | Scanned PDFs | Instant loading | Smart caching

[![Features](https://img.shields.io/badge/Features-12+-blue)](#features)
[![OCR](https://img.shields.io/badge/OCR-Supported-green)](#ocr-processing)
[![Cache](https://img.shields.io/badge/Smart-Caching-orange)](#intelligent-caching)
[![Interface](https://img.shields.io/badge/Interface-CLI%20%7C%20Web-purple)](#quick-start)

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-format Support** | PDF, DOCX, TXT files with automatic format detection |
| 🔍 **Smart Search** | Hybrid TF-IDF + exact matching for precise retrieval |
| 🧾 **OCR Processing** | Extract text from scanned PDFs automatically |
| ⚡ **Instant Loading** | Smart caching: 10+ min → 2-3 seconds |
| 🧠 **AI Responses** | Google Gemini with inline citations |
| � **Dual Interface** | CLI + Beautiful Streamlit web app |

## 🚀 Quick Start

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Configure API Key
```bash
# Create .env file
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

### 3️⃣ Add Documents
```bash
# Copy your files to data folder
cp your_documents.pdf data/
```

### 4️⃣ Launch Application

**🌐 Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
```

**💻 Command Line**
```bash
python cli_chatbot.py
```

**⚡ For Large Scanned PDFs**
```bash
# Pre-process for instant loading
python cache_builder.py
```

## 📁 Project Structure

```
📦 Chatbot/
├── 📂 data/                     # 📄 Your documents go here
├── 📂 cache/                    # 💾 Auto-generated cache files
├── 🔐 .env                      # 🔑 API configuration
├── 📋 requirements.txt          # 📦 Dependencies
├── 🏗️  document_loader.py       # 📄 Enhanced loading + OCR
├── 🧠 gemini_wrapper.py         # 🤖 AI integration
├── 🔍 retriever.py             # 📊 Smart search engine
├── 💻 cli_chatbot.py           # 🖥️  Command line interface
├── 🌐 streamlit_app.py         # 🎨 Web interface
├── ⚡ cache_builder.py         # 🏗️  Pre-process large files
└── 🛠️  ocr_configurator.py     # 🔧 OCR configuration tool
```

## 🔧 Components

### Document Loader (`document_loader.py`)
- Loads PDF, DOCX, and TXT files from the `data/` folder
- **OCR Support**: Automatically processes scanned PDFs using Tesseract OCR
- **Smart Caching**: Saves OCR results to cache for instant future loading
- **Hybrid Processing**: Regular text extraction + OCR fallback for scanned documents
- Handles multiple file formats with comprehensive error handling

### Gemini API Wrapper (`gemini_wrapper.py`)
- Provides easy interface to Google Gemini API
- Handles context-aware question answering with inline citations
- Manages API authentication and error handling

### Document Retriever (`retriever.py`)
- Uses TF-IDF vectorization for document similarity
- Finds most relevant document chunks for queries
- Returns ranked results with similarity scores

### CLI Interface (`cli_chatbot.py`)
- Interactive command-line chat experience
- Commands: help, list, history, clear, quit
- Shows relevant documents for each query

### Web Interface (`streamlit_app.py`)
- Beautiful web-based chat interface
- Document management sidebar
- Chat history with source attribution
- Real-time interaction

## � Usage Examples

### CLI Commands
| Command | Action | Example |
|---------|--------|---------|
| `help` | Show commands | `💬 You: help` |
| `list` | Show documents | `💬 You: list` |
| `refresh` | Reload files | `💬 You: refresh` |
| `summarize` | Generate summary | `💬 You: summarize` |
| `history` | View chat log | `💬 You: history` |

### Example Questions
- *"What's the main topic of the documents?"*
- *"Summarize the key points from the PDFs"*
- *"What happened in March 2021?"* → **[1] company.txt**
- *"Compare the backend technologies"* → **[1] tech_docs.pdf [2] web_dev.txt**

## 🔄 Adding New Documents

**📂 Simply drop files in `data/` folder**

| Interface | Steps |
|-----------|-------|
| 🌐 **Web** | 1. Add files to `data/` → 2. Click 🔄 Refresh button |
| 💻 **CLI** | 1. Add files to `data/` → 2. Type `refresh` |

**Supported formats**: `.pdf` `.docx` `.txt` (including scanned PDFs)

## 🎯 Core Features

<details>
<summary><strong>⚡ Instant Loading with OCR Caching</strong></summary>

**Problem Solved**: Large scanned PDFs that took 10+ minutes to load now load in seconds!

| Document Type | First Load | Subsequent Loads |
|---------------|------------|------------------|
| 📄 Text PDF | ~2 seconds | ~1 second |
| 🖼️ Scanned PDF (215 pages) | ~10-15 minutes | **~2-3 seconds** ⚡ |

**How it works**:
- 🔍 Auto-detects scanned vs text PDFs
- 💾 Caches OCR results with file hash verification
- 🔄 Only re-processes when files actually change
- 📊 Real-time progress tracking
</details>

<details>
<summary><strong>📖 Smart Citations</strong></summary>

**Automatic source citations** in every response:

```
In March 2021, SynthoTech Solutions released PipeStream v1.0 [1]. 
The system supports Python and Node.js backends [2].

References:
[1] company_report.pdf
[2] technical_docs.txt
```

✅ **Numbered references** | ✅ **Inline citations** | ✅ **Multi-source tracking**
</details>

<details>
<summary><strong>🔍 Hybrid Search Engine</strong></summary>

**Combines multiple search strategies**:
- 🎯 **Exact keyword matching** for precise queries
- 📊 **TF-IDF similarity** for semantic search  
- 📅 **Date-aware processing** for temporal queries
- 🏆 **Smart scoring** prioritizes best matches
</details>

## 🔧 How It Works

1. **📂 Document Loading** → Scans `data/` folder for supported files
2. **🔍 Text Extraction** → Regular text + OCR for scanned PDFs  
3. **📊 Smart Indexing** → TF-IDF vectors + exact match patterns
4. **🤖 AI Processing** → Gemini API with context + citations

## ⚙️ Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `GEMINI_API_KEY` | Google Gemini API key | *Required* |
| Max Documents | Context size for AI | 3 |
| Similarity Threshold | Relevance cutoff | 0.05 |

## 🛠️ Troubleshooting

<details>
<summary><strong>🔧 Common Issues & Solutions</strong></summary>

| Problem | Solution |
|---------|----------|
| � **API Key Error** | Check `.env` file with valid `GEMINI_API_KEY` |
| 📄 **No Documents** | Verify files in `data/` folder (.pdf, .docx, .txt) |
| 📦 **Import Errors** | Run `pip install -r requirements.txt` |
| 🐌 **Slow Loading** | Use `python cache_builder.py` for large files |
| 🖼️ **Scanned PDFs** | ✅ **Fully supported** with automatic OCR |

**📊 Performance Tips**:
- Use cache builder for 100+ page PDFs
- First OCR processing: ~10-15 minutes  
- All subsequent loads: ~2-3 seconds ⚡
</details>

## 🛠️ Troubleshooting

### Common Issues

1. **API Key Error**
   - Ensure `.env` file exists with valid `GEMINI_API_KEY`
   - Check API key permissions and quota

2. **No Documents Found**
   - Verify documents are in the `data/` folder
   - Check file formats (.pdf, .docx, .txt)
   - Ensure files are not corrupted

3. **Import Errors**
   - Run: `pip install -r requirements.txt`
   - Ensure all dependencies are installed

4. **Document Loading Errors**
   - Check file permissions
   - Verify file encoding (especially for TXT files)
   - Try with smaller files first

5. **AI Asking for Document Text**
   - Try refreshing documents using the refresh button
   - Clear browser cache and reload the page
   - Check that documents are properly loaded in the sidebar

6. **Scanned PDF Files**
   - ✅ **Fully Supported**: The system now automatically processes scanned PDFs using OCR
   - **First-time Processing**: Large scanned PDFs may take 10-15 minutes to process initially
   - **Instant Loading**: After initial OCR processing, files load in seconds from cache
   - **Cache Builder**: Use `python cache_builder.py` to pre-process large documents
   - **Progress Tracking**: Real-time updates show OCR processing progress
   - **Full Document Access**: Complete text extraction from all pages (not just samples)

7. **Slow Loading Times**
   - **For Large Scanned PDFs**: Use the cache builder to pre-process documents
   - **Run**: `python cache_builder.py` to build cache for all documents
   - **Result**: App will load instantly after cache is built
   - **Cache Location**: OCR results stored in `cache/` directory

## 📦 Dependencies

### Core Requirements
```bash
pip install google-generativeai python-dotenv streamlit PyPDF2 python-docx scikit-learn numpy
```

### OCR Requirements (for scanned PDFs)
```bash
pip install PyMuPDF pytesseract Pillow
```

**System Requirements**: Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- 🪟 **Windows**: Download installer from GitHub
- 🐧 **Linux**: `sudo apt-get install tesseract-ocr`  
- 🍎 **macOS**: `brew install tesseract`

---

<div align="center">

## 🎉 Ready to Chat with Your Documents?

**[⬆️ Quick Start](#quick-start)** | **[📖 Examples](#usage-examples)** | **[�️ Troubleshooting](#troubleshooting)**

*Built with ❤️ using Google Gemini AI*

</div>
