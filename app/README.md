# RAG Multi-PDF Chat App

This folder contains several Tkinter-based GUI variants of a Retrieval-Augmented Generation (RAG) application for chatting with multiple PDFs.

## Features (latest app_6.py)
- Select and load multiple PDF files.
- Automatic text splitting and FAISS vector index building.
- Ask natural language questions, get answers with sources.
- Chat history maintained per session.
- Right-hand panel shows retrieved sources, cosine scores, and ORIGINAL vs REFINED spans (with correlation-driven optimization).
- Manual chunk editor before indexing.
- Thread-safe Tkinter dialogs (no “main loop” errors when prompting from background threads).
- EXE-compatible (PyInstaller support with Tcl/Tk and FAISS DLL fixes, one-dir and one-file builds supported).

## File Overview
- app.py – Initial baseline with basic multi-PDF chat.
- app_2.py – Added boundary-refine retrieval (score before/after).
- app_3.py – Minimal GUI refactor (sources/snippets only).
- app_4.py – Full-featured version (boundary-refine + provenance panel).
- app_5.py – Added correlation-driven optimization to refinements; EXE build confirmed (no console window).
- app_6.py – Latest version, integrates all features with thread-safe UI and resilient API-key handling.
- app_mo.py – Modular/lightweight version for simplified builds.

## How to Run
```bash
cd Rag-multipdf/app
python app_6.py
