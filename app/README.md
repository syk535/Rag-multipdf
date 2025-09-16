# RAG Multi-PDF Chat App

This folder contains several Tkinter-based GUI variants of a Retrieval-Augmented Generation (RAG) application for chatting with multiple PDFs.

## Features (latest app_4.py)
- Select and load multiple PDF files.
- Automatic text splitting and FAISS vector index building.
- Ask natural language questions, get answers with sources.
- Chat history maintained per session.
- Right-hand panel shows retrieved sources, cosine scores, and ORIGINAL vs REFINED spans.
- Manual chunk editor before indexing.
- EXE-compatible (PyInstaller support with Tcl/Tk and faiss DLL fixes).

## File Overview
- app.py – Initial baseline with basic multi-PDF chat.
- app_2.py – Added boundary-refine retrieval (score before/after).
- app_3.py – Minimal GUI refactor (sources/snippets only).
- app_4.py – Current full-featured version (boundary-refine + provenance panel).
- app_mo.py – Modular/lightweight version for simplified builds.

## How to Run
```bash
cd Rag-multipdf/app
python app_4.py
