#!/usr/bin/env python3
"""
export_for_notebooklm.py

Crawls the entire repository, ignoring irrelevant directories and files,
and concatenates source code into a single massive text file.
"""

import os
from pathlib import Path

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = PROJECT_ROOT / "notebookllm_source.txt"

# Files to include
ALLOWED_EXTENSIONS = {".py", ".js", ".md", ".json", ".yml", ".yaml"}

# Directories to explicitly ignore (won't be crawled)
IGNORE_DIRS = {
    ".git", 
    "__pycache__", 
    "venv", 
    ".venv", 
    "node_modules", 
    ".idea", 
    ".vscode", 
    "runs", 
    "datasets", 
    "yolo26n.mlpackage", 
    "yolo26n_int8.mlpackage",
    "ml_image_similarity"
}

# Specific files to ignore
IGNORE_FILES = {
    "yolo26n.pt", 
    "notebookllm_source.txt",
    "package-lock.json"
}

def main():
    print(f"🔍 Starting codebase crawl rooted at: {PROJECT_ROOT}")
    print(f"📦 Assembling NotebookLM export to: {OUTPUT_FILE}")
    
    files_processed = 0
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # Walk through the directory tree
        for root, dirs, files in os.walk(PROJECT_ROOT):
            # Modify dirs in-place to prune ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS and not d.startswith('.')]
            
            # Sort files for deterministic output
            files.sort()
            
            for file in files:
                if file in IGNORE_FILES or file.startswith('.'):
                    continue
                    
                file_path = Path(root) / file
                
                # Check extension
                if file_path.suffix.lower() in ALLOWED_EXTENSIONS:
                    # Calculate relative path from project root
                    rel_path = file_path.relative_to(PROJECT_ROOT)
                    
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            content = infile.read()
                            
                        # Write the header and content
                        outfile.write(f"\n======= FILE: {rel_path} =======\n\n")
                        outfile.write(content)
                        outfile.write("\n")
                        
                        files_processed += 1
                        
                    except Exception as e:
                        print(f"⚠️  Skipping {rel_path} due to read error: {e}")
                        
    print(f"✅ Export complete. Successfully merged {files_processed} files.")
    print(f"📄 Output saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
