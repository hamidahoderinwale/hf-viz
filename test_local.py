"""
Quick test script to verify the application works locally.
Run this before deploying to Hugging Face Spaces.
"""
import sys
from app import create_interface

if __name__ == "__main__":
    print("Creating interface...")
    demo = create_interface()
    print("Launching demo...")
    demo.launch(share=False, server_name="127.0.0.1", server_port=7860)

