# Book Library Analyzer

This Python script analyzes a directory of e-books to help you understand your collection.

## Features

1.  **Library Characterization**:

      * Counts total files.
      * Counts files by type (`.pdf`, `.epub`, `.mobi`, `.zip`).
      * Calculates the total size of your library.

2.  **Metadata Extraction**:

      * Reads metadata (Title, Author) directly from `.epub` and `.pdf` files.
      * Uses filenames as a fallback for `.mobi`, `.zip`, and files with missing metadata.

3.  **Duplicate Detection**:

      * Normalizes titles (removes punctuation, case, etc.) to find logical duplicates.
      * Generates a report of all titles that appear in **more than one folder**, helping you clean up your collection.

4.  **Topical Analysis (Optional)**:

      * Connects to a local [Ollama](https://ollama.com/) instance.
      * Sends each unique title and author to an LLM (currently defaulted to **granite4:micro-h**) to get a subject/genre classification.
      * Provides a final report on the topical coverage of your library (e.g., "Science Fiction: 50", "History: 20").

## Requirements

You must have Python 3 installed, along with a few packages:

```bash
pip install PyMuPDF ebooklib requests
