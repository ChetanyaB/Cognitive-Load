import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Importing app...")
from app.gradio_app import build_app

print("Building app...")
demo = build_app()

print(f"demo type: {type(demo)}")

if demo is None:
    print("ERROR: build_app() returned None.")
    sys.exit(1)

print("Launching...")
demo.launch(share=False)