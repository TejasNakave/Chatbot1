# 🚀 Deployment Guide

## Quick Deploy to Linux Server

### 1. Server Setup
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv tesseract-ocr -y
```

### 2. Clone & Install
```bash
git clone https://github.com/TejasNakave/Chatbot1.git
cd Chatbot1
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
mkdir -p data cache
```

### 4. Upload Documents
```bash
# Copy your documents to data/ folder
scp your_docs.pdf user@server:/path/to/Chatbot1/data/
```

### 5. Run Application
```bash
# Start the web app
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0

# Or use the launcher
python launcher.py
```

### 6. Access
- **Local**: http://localhost:8501
- **Server**: http://your-server-ip:8501

## Production Setup

### Systemd Service
```bash
sudo tee /etc/systemd/system/chatbot.service << EOF
[Unit]
Description=Document AI Chatbot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/path/to/Chatbot1
Environment=PATH=/path/to/Chatbot1/venv/bin
ExecStart=/path/to/Chatbot1/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable chatbot
sudo systemctl start chatbot
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Requirements

- **Python**: 3.8+
- **RAM**: 2GB+ recommended
- **Storage**: 1GB+ for cache
- **API**: Google Gemini API key

## File Structure

```
Chatbot1/
├── streamlit_app.py     # Main web application
├── document_loader.py   # Document processing
├── gemini_wrapper.py    # AI integration
├── retriever.py         # Search engine
├── launcher.py          # Quick launcher
├── requirements.txt     # Dependencies
├── data/               # Your documents
├── cache/              # Processed cache
└── .env               # API keys
```