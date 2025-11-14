#!/usr/bin/env python3

import sys
import re
import json
import requests
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, Dict, Any

# We use optional imports for PyMuPDF and EbookLib to handle cases where they aren't installed.
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    import ebooklib
    from ebooklib import epub
except ImportError:
    ebooklib = None
    epub = None

# --- Configuration ---

# Set to True to run the (slow) subject matter analysis
RUN_OLLAMA_ANALYSIS = True 
OLLAMA_URL = "http://localhost:11434/api/generate"
# Change this to your preferred model (e.g., "granite4:1b", "llama3")
OLLAMA_MODEL = "granite4:micro-h" 
# Max *bytes* of text to send to Ollama. 32768 is a good default.
OLLAMA_CONTEXT_SIZE = 65536
# Maximum number of retries for Ollama requests
MAX_RETRIES = 3
INITIAL_BACKOFF = 2 

# --- Helper Functions ---

def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string (KB, MB, GB)."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    for unit in ['KB', 'MB', 'GB', 'TB']:
        size_bytes /= 1024.0
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
    return f"{size_bytes:.2f} PB"

def calculate_hash(file_path: Path) -> str:
    """Calculates the SHA256 hash of a file for exact duplicate checking."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            # Read and update hash string in chunks
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        # On failure (e.g., permission error), return a unique error string
        return f"HASH_ERROR_{str(e)}"

# --- Metadata Extraction ---

def normalize_title(title: str) -> str:
    """Cleans a title string for consistent matching, preserving edition info."""
    if not title:
        return "Unknown Title"
    
    title = title.lower()
    
    # Replace common separators with spaces to split compound filenames
    # This turns "python_book_3rd" into "python book 3rd"
    title = re.sub(r'[\._\-]', ' ', title)
    
    # Remove any text inside parentheses or brackets that might be junk tags (like [retail] or (sample))
    # Note: this is now less aggressive to try and keep edition info like (3rd Edition)
    # Reverting to less aggressive stripping that was in the previous version, 
    # but ensuring we normalize what remains.
    
    # Remove all remaining punctuation
    title = re.sub(r'[^\w\s]', '', title)
    
    # Collapse whitespace (multiple spaces to one)
    title = re.sub(r'\s+', ' ', title).strip()
    
    return title or "Unknown Title"

def get_epub_meta(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extracts title and author from EPUB files."""
    if not ebooklib:
        return None, None, "EbookLib not installed."
        
    try:
        book = epub.read_epub(str(path))
        title = book.get_metadata('dc', 'title')
        author = book.get_metadata('dc', 'creator')
        
        # Extract first title and author, clean up
        t = title[0][0].strip() if title else None
        a = author[0][0].strip() if author else None
        
        return t, a, None
    except Exception as e:
        return None, None, f"'Bad Zip file' or other EPUB error: {e}"

def get_pdf_meta(path: Path) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extracts title and author from PDF files using PyMuPDF."""
    if not fitz:
        return None, None, "PyMuPDF not installed."

    try:
        with fitz.open(str(path)) as doc:
            meta = doc.metadata
            title = meta.get('title')
            author = meta.get('author')
            
            # Clean up: PyMuPDF can return titles like 'b'' or 'Untitled'
            t = title.strip() if title and title.lower() not in ('', 'untitled', 'microsoft word') else None
            a = author.strip() if author else None
            
            return t, a, None
    except Exception as e:
        return None, None, f"PDF Error '{path.name}': {e}"

# --- Ollama Classification (Helpers) ---

def extract_text_from_epub(path: Path, limit_bytes: int) -> str:
    """Extracts plain text from an EPUB, up to a byte limit."""
    if not ebooklib: return ""
    try:
        text = ""
        book = epub.read_epub(str(path))
        
        # Iterate over document items (chapters, main content)
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            content = item.get_content().decode('utf-8', 'ignore')
            
            # Basic regex to strip HTML tags and clean up
            plain_text = re.sub(r'<[^>]+>', ' ', content)
            plain_text = re.sub(r'\s+', ' ', plain_text).strip()
            
            # Check byte size before appending
            current_byte_size = len(text.encode('utf-8'))
            if current_byte_size >= limit_bytes:
                break
                
            plain_text_bytes = plain_text.encode('utf-8')
            
            if current_byte_size + len(plain_text_bytes) > limit_bytes:
                remaining_bytes = limit_bytes - current_byte_size
                
                # Decode remaining bytes to avoid cutting in the middle of a character
                text += plain_text_bytes[:remaining_bytes].decode('utf-8', 'ignore')
                break
                
            text += plain_text + "\n"
        
        return text
    except Exception as e:
        print(f"  [!] Warning: Could not extract text from EPUB {path.name}: {e}", file=sys.stderr)
        return ""

def extract_text_from_pdf(path: Path, limit_bytes: int) -> str:
    """Extracts plain text from a PDF, up to a byte limit."""
    if not fitz: return ""
    try:
        text = ""
        with fitz.open(str(path)) as doc:
            for page in doc:
                page_text = page.get_text("text")
                
                current_byte_size = len(text.encode('utf-8'))
                if current_byte_size >= limit_bytes:
                    break
                    
                page_text_bytes = page_text.encode('utf-8')

                # Check byte size before appending
                if current_byte_size + len(page_text_bytes) > limit_bytes:
                    remaining_bytes = limit_bytes - current_byte_size
                    # Decode remaining bytes to avoid cutting in the middle of a character
                    text += page_text_bytes[:remaining_bytes].decode('utf-8', 'ignore')
                    break
                
                text += page_text + "\n"
        
        return text
    except Exception as e:
        print(f"  [!] Warning: Could not extract text from PDF {path.name}: {e}", file=sys.stderr)
        return ""

def get_subject_from_ollama(title: str, author: str, context_text: str) -> str:
    """
    Uses a local Ollama instance to classify a book by subject/genre,
    using the book's text for context, with retry and improved error handling.
    """
    if context_text:
        prompt = (
            f"Please classify the following book into a single, primary subject or genre. "
            f"Use the provided Table of Contents and introductory text for context.\n\n"
            f"Title: {title}\n"
            f"Author: {author}\n\n"
            f"Context Text:\n---\n{context_text}\n---\n\n"
            f"Respond with *only* the subject name and nothing else "
            "(e.g., 'Science Fiction', 'History', 'Biography', 'Python Programming', 'Fantasy')."
        )
    else:
        # Fallback if no text could be extracted
        prompt = (
            f"Given the book title '{title}' and author '{author}', "
            "classify it into a single, primary subject or genre. "
            "Respond with *only* the subject name and nothing else "
            "(e.g., 'Science Fiction', 'History', 'Biography', 'Python Programming', 'Fantasy')."
        )

    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 50, # Keep output concise
            "num_ctx": OLLAMA_CONTEXT_SIZE # Use the configured context size
        }
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status() # Raises HTTPError for bad status codes (4xx or 5xx)
            
            # Success
            result = response.json()
            # Clean up the response text
            subject = result.get('response', '').strip()
            # Remove any quotes the LLM might incorrectly add
            return subject.strip('"').strip("'")
            
        except requests.exceptions.HTTPError as e:
            last_error = e
            if e.response.status_code == 404:
                # 404 is critical and almost always a missing model, so we don't retry network errors
                error_message = (
                    f"404 Not Found. This means the model '{OLLAMA_MODEL}' is not available in your "
                    f"Ollama installation. Please run: 'ollama pull {OLLAMA_MODEL}' in your terminal."
                )
                print(f"  [!] CRITICAL OLLAMA ERROR for '{title}': {error_message}", file=sys.stderr)
                return "Classification Failed: Missing Model"
            
            print(f"  [!] Warning: HTTP Error on attempt {attempt + 1}/{MAX_RETRIES} for '{title}': {e}", file=sys.stderr)
            
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"  [!] Warning: Connection Error on attempt {attempt + 1}/{MAX_RETRIES} for '{title}': {e}", file=sys.stderr)

        # Apply exponential backoff before retrying
        if attempt < MAX_RETRIES - 1:
            wait_time = INITIAL_BACKOFF * (2 ** attempt)
            print(f"    - Retrying in {wait_time} seconds...", file=sys.stderr)
            import time
            time.sleep(wait_time)

    # If all retries fail
    print(f"  [!] Failed to classify '{title}' after {MAX_RETRIES} attempts. Last error: {last_error}", file=sys.stderr)
    return "Classification Failed: Connection Error"


# --- Main Analysis ---

def analyze_books(start_path: str):
    """
    Main function to analyze the book directory, build the database,
    detect duplicates, and optionally classify subjects using Ollama.
    """
    
    start_path_obj = Path(start_path)
    if not start_path_obj.is_dir():
        print(f"Error: Path '{start_path}' is not a valid directory.", file=sys.stderr)
        return

    # --- Pass 1: Scan and Build Database ---
    
    print(f"\n--- Pass 1: Scanning Directory and Building Database ---")
    
    file_counts = defaultdict(int)
    total_size = 0
    
    # Key: normalized title, Value: {title, authors, files:[{path, format, size, hash, metadata_source, metadata_error, is_duplicate}], subject}
    books_db: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'title': None, 
        'authors': set(), 
        'files': [], 
        'subject': None
    })
    
    # List to collect all metadata errors for the final report
    all_metadata_errors = [] 

    # Supported file extensions
    SUPPORTED_EXTENSIONS = ['.pdf', '.epub', '.mobi', '.zip']

    for item in start_path_obj.rglob('*'):
        if item.is_file():
            ext = item.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                continue

            file_counts['TOTAL'] += 1
            file_counts[ext] += 1
            
            try:
                file_size = item.stat().st_size
                total_size += file_size
            except Exception:
                file_size = 0 # Cannot get size

            title = None
            author = None
            metadata_error = None
            metadata_source = "Filename"
            
            # --- Attempt Metadata Extraction ---
            if ext == '.epub':
                title, author, metadata_error = get_epub_meta(item)
            elif ext == '.pdf':
                title, author, metadata_error = get_pdf_meta(item)
            
            # Handle metadata extraction results
            if metadata_error:
                all_metadata_errors.append((item.name, ext, metadata_error))
            
            if title is None:
                # Fallback: use filename as title
                raw_title = item.stem 
                title = raw_title
            else:
                metadata_source = ext.upper() # Source is PDF or EPUB metadata

            # --- Normalization and Storage ---
            norm_title = normalize_title(title)
            
            if books_db[norm_title]['title'] is None:
                books_db[norm_title]['title'] = title # Store the best raw title found
                
            if author:
                # Add author to the set of authors for this unique book
                books_db[norm_title]['authors'].add(author)

            # Store file details (hash will be calculated in Pass 2 if needed)
            books_db[norm_title]['files'].append({
                'path': str(item),
                'format': ext,
                'size': file_size,
                'hash': None, # To be filled in Pass 2
                'metadata_source': metadata_source,
                'metadata_error': metadata_error
            })

    num_unique_titles = len(books_db)
    print(f"Scan complete. Processed {file_counts['TOTAL']} files. Found {num_unique_titles} unique titles.")


    # --- Pass 2: Duplicate Detection and Hashing ---

    print(f"\n--- Pass 2: Identifying Duplicates and Calculating Hashes ---")
    
    # Find titles that appear in more than one location (potential duplicates)
    potential_duplicates = {
        norm_title: info for norm_title, info in books_db.items() 
        if len(set(Path(f['path']).parent for f in info['files'])) > 1 
    }
    
    # Track files that need hashing (only those involved in potential duplicates)
    files_to_hash = []
    for info in potential_duplicates.values():
        files_to_hash.extend(info['files'])
        
    print(f"Found {len(potential_duplicates)} titles that exist in multiple folders.")
    print(f"Calculating SHA256 hashes for {len(files_to_hash)} files involved in duplicates...")

    # Calculate hashes only for relevant files
    for file_data in files_to_hash:
        file_data['hash'] = calculate_hash(Path(file_data['path']))

    # --- Pass 3: Ollama Classification ---

    if RUN_OLLAMA_ANALYSIS:
        print(f"\n--- Pass 3: Classifying Subjects (Ollama) ---")
        print(f"Using Model: {OLLAMA_MODEL}, Context Limit: {human_readable_size(OLLAMA_CONTEXT_SIZE)}")
        print(f"This may take a long time. Analyzing {num_unique_titles} unique titles...")
        
        # Iterate over unique titles (keys of books_db)
        for i, (norm_title, info) in enumerate(books_db.items()):
            print(f"  [{i+1}/{num_unique_titles}] Classifying '{norm_title}'...")
            
            author_str = ", ".join(info['authors']) or "Unknown"
            context_text = ""

            # Find a PDF or EPUB file to extract text from
            preferred_file = None
            for f in info['files']:
                ext = f['format']
                if ext == '.epub': # Prefer EPUB
                    preferred_file = f
                    break
                elif ext == '.pdf':
                    preferred_file = f # Fallback to PDF
            
            if preferred_file:
                path = Path(preferred_file['path'])
                print(f"    - Extracting text from {path.name} for context...")
                if preferred_file['format'] == '.epub':
                    context_text = extract_text_from_epub(path, OLLAMA_CONTEXT_SIZE)
                elif preferred_file['format'] == '.pdf':
                    context_text = extract_text_from_pdf(path, OLLAMA_CONTEXT_SIZE)

            subject = get_subject_from_ollama(norm_title, author_str, context_text)
            
            # Store the result back in the database
            info['subject'] = subject

    # --- Report Generation ---
    
    # Report 1: Library Characterization
    print("\n\n=======================================================")
    print("Report 1: Library Characterization")
    print("=======================================================")
    print(f"Total Files Scanned: {file_counts['TOTAL']}")
    print(f"Total Unique Titles: {num_unique_titles}")
    print(f"Total Library Size:  {human_readable_size(total_size)}")
    print("\nFile Format Counts:")
    for ext in SUPPORTED_EXTENSIONS:
        print(f"  {ext.upper()}: {file_counts[ext]}")
    
    # Report 2: Duplicate Title Report
    if potential_duplicates:
        print("\n\n=======================================================")
        print(f"Report 2: Duplicate Titles ({len(potential_duplicates)} Unique Titles)")
        print("=======================================================")
        
        for norm_title, info in potential_duplicates.items():
            print(f"\n[Title] {info['title']} (Normalized: {norm_title})")
            print(f"    Authors: {', '.join(info['authors']) or 'Unknown'}")
            
            # Group files by their hash
            hash_groups = defaultdict(list)
            for f in info['files']:
                # Filter out files that failed to hash
                if f['hash'] and not f['hash'].startswith("HASH_ERROR"):
                    hash_groups[f['hash']].append(f)
            
            
            # Check for files that couldn't be hashed
            hash_errors = [f for f in info['files'] if f['hash'].startswith("HASH_ERROR")]
            if hash_errors:
                 print(f"    [!] WARNING: Could not hash {len(hash_errors)} files due to errors (e.g., permissions).")


            # Process grouped hashes
            for file_hash, files in hash_groups.items():
                if len(files) > 1:
                    # EXACT MATCHES (Same content hash)
                    print(f"\n  [EXACT MATCHES] ({len(files)} identical files, Hash: {file_hash[:10]}...)")
                    for f in files:
                        path_obj = Path(f['path'])
                        size_str = human_readable_size(f['size'])
                        print(f"    - {path_obj.parent} / {path_obj.name} ({f['format']}, {size_str})")
                
            # SIMILAR TITLES (Different hashes)
            unique_hashes = list(hash_groups.keys())
            if len(unique_hashes) > 1:
                print(f"\n  [SIMILAR TITLES] ({len(unique_hashes)} different versions/formats)")
                
                # List all files that are not part of an exact match group (i.e., unique versions)
                all_files_in_groups = sum([g for g in hash_groups.values()], [])
                unique_versions = [f for f in info['files'] if f not in all_files_in_groups]
                
                # Iterate through all files and report them by version
                for f in info['files']:
                    path_obj = Path(f['path'])
                    size_str = human_readable_size(f['size'])
                    hash_indicator = f['hash'][:10] + '...' if f['hash'] else 'N/A'
                    print(f"    - {path_obj.parent} / {path_obj.name} ({f['format']}, {size_str}, Hash: {hash_indicator})")

    else:
        print("\n\n=======================================================")
        print("Report 2: Duplicate Titles (Not Run)")
        print("=======================================================")
        print("Set RUN_OLLAMA_ANALYSIS = True to enable subject classification.")

    # Report 3: Topical Coverage
    if RUN_OLLAMA_ANALYSIS:
        print("\n\n=======================================================")
        print("Report 3: Topical Coverage by Subject")
        print("=======================================================")
        
        subject_counts = defaultdict(int)
        for info in books_db.values():
            subject_counts[info['subject'] or "Unclassified"] += 1
            
        # Sort subjects by count descending
        sorted_subjects = sorted(subject_counts.items(), key=lambda item: item[1], reverse=True)
        
        for subject, count in sorted_subjects:
            print(f"  {subject}: {count}")

    # Report 4: Metadata Extraction Errors
    if all_metadata_errors:
        print("\n\n=======================================================")
        print(f"Report 4: Metadata Extraction Errors ({len(all_metadata_errors)} total)")
        print("=======================================================")
        print("These files failed to yield title/author from internal metadata, so the filename was used.")
        print("\nSample Errors (First 10):")
        for i, (name, ext, error) in enumerate(all_metadata_errors[:10]):
            print(f"  [{ext.upper()}] {name}: {error}")
        if len(all_metadata_errors) > 10:
            print(f"  ...and {len(all_metadata_errors) - 10} more.")


def main():
    """Handles command line arguments and runs the analyzer."""
    if len(sys.argv) > 1:
        start_path = sys.argv[1]
    else:
        start_path = input("Enter the path to your book library: ")

    analyze_books(start_path)

if __name__ == "__main__":
    main()
