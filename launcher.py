#!/usr/bin/env python3
"""
Launcher script for the Document-Based Chatbot
"""

import os
import sys
import subprocess


def run_command(command):
    """Run a command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")


def main():
    """Main launcher menu."""
    while True:
        print("\n" + "="*50)
        print("    🤖 Document-Based Chatbot Launcher")
        print("="*50)
        print("\nSelect an option:")
        print("1. 💻 Run CLI Chatbot")
        print("2. 🌐 Run Streamlit Web App")
        print("3. 📄 Test Document Loader")
        print("4. 🧠 Test Gemini API")
        print("5. 📊 View Project Structure")
        print("6. ❌ Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\n🚀 Starting CLI Chatbot...")
                run_command("python cli_chatbot.py")
                
            elif choice == "2":
                print("\n🌐 Starting Streamlit Web App...")
                print("The web app will open in your browser...")
                run_command("streamlit run streamlit_app.py")
                
            elif choice == "3":
                print("\n📄 Testing Document Loader...")
                run_command("python document_loader.py")
                input("\nPress Enter to continue...")
                
            elif choice == "4":
                print("\n🧠 Testing Gemini API...")
                run_command("python gemini_wrapper.py")
                input("\nPress Enter to continue...")
                
            elif choice == "5":
                print("\n📊 Project Structure:")
                print("Chatbot/")
                print("├── data/                    # Place your documents here")
                print("├── .env                     # API key configuration")
                print("├── requirements.txt         # Python dependencies")
                print("├── document_loader.py       # Document loading functions")
                print("├── gemini_wrapper.py        # Gemini API wrapper")
                print("├── retriever.py            # Document retrieval system")
                print("├── cli_chatbot.py          # Command-line interface")
                print("├── streamlit_app.py        # Web interface")
                print("├── launcher.py             # This launcher script")
                print("└── README.md               # Documentation")
                input("\nPress Enter to continue...")
                
            elif choice == "6":
                print("\n👋 Goodbye! Thanks for using the Document Chatbot!")
                break
                
            else:
                print("❌ Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Document Chatbot!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    # Change to the script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
