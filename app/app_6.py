# -*- coding: utf-8 -*-
"""
RAG Multi-PDF Chat (Tkinter GUI, single file)
- Stage 1: Manual chunk review / keep/discard / inline edit (ChunkEditor).
- Stage 2: At query time, retrieve top-K, then refine chunk boundaries:
    (A) Simple expand: +/- N characters around the original chunk span;
    (B) Optional sliding optimization: within +/- N chars, slide a fixed-size
        window to maximize cosine similarity with the query (argmax).
- Right panel shows: file, page, ORIGINAL text, REFINED text, score_before/after, Δ, span changes.

Key robustness:
- No tkinter messagebox in worker threads (prevents "main thread is not in main loop").
- When frozen (PyInstaller --onefile), looks for .env / OPENAI_API_KEY.txt next to the EXE.
- If OPENAI_API_KEY is still missing, prompts once via Tk simpledialog (main thread).
- Includes tiktoken_ext.openai_public import so encodings (cl100k_base) are registered.

Requirements (pip):
  pyinstaller python-dotenv tiktoken langchain langchain-openai langchain-community faiss-cpu openai pypdf
"""

# --- EXE-compatible path & API key loader ---
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Make sure OpenAI encodings are registered (important for tiktoken)
import tiktoken_ext.openai_public  # noqa: F401


def _pin_tcl_tk_for_pyinstaller():
    """When frozen by PyInstaller, let Tk know where the bundled Tcl/Tk lives."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base and os.path.isdir(base):
            tcl = os.path.join(base, "tcl", "tcl8.6")
            tk = os.path.join(base, "tcl", "tk8.6")
            os.environ.setdefault("TCL_LIBRARY", tcl)
            os.environ.setdefault("TK_LIBRARY", tk)


_pin_tcl_tk_for_pyinstaller()


def APP_DIR() -> Path:
    """
    Return the directory where app resources live.
    - When frozen by PyInstaller, it's the folder next to the executable.
    - When running as a script, it's the folder of this file.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


APP_PATH = APP_DIR()

# Prefer reading .env / OPENAI_API_KEY.txt next to the exe (or this script)
dotenv_file = APP_PATH / ".env"
if dotenv_file.exists():
    load_dotenv(dotenv_file)
else:
    load_dotenv()  # Fallback: load .env from CWD if present

# Fallback: read OPENAI_API_KEY from a plaintext file next to the exe/script
if not os.getenv("OPENAI_API_KEY"):
    key_txt = APP_PATH / "OPENAI_API_KEY.txt"
    if key_txt.exists():
        os.environ["OPENAI_API_KEY"] = key_txt.read_text(encoding="utf-8").strip()

# ---- std libs & typing ----
import math
import threading
import queue
from typing import List, Tuple, Optional

# ---- GUI ----
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ---- LangChain / OpenAI ----
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.schema import Document


# =========================== Utilities & RAG helpers ===========================

def pick_pdfs(parent) -> List[str]:
    """Open file dialog for PDFs."""
    return list(
        filedialog.askopenfilenames(
            parent=parent, title="Select PDF files", filetypes=[("PDF files", "*.pdf")]
        )
    )


def load_documents(pdf_paths: List[str]) -> Tuple[List[Document], List[str]]:
    """
    Load PDFs as page-level Documents.
    Ensure metadata contains 'source' (filename) and 'page' (0-based).
    Return (docs, warnings) so UI thread can decide how/when to show dialogs.
    """
    docs = []
    warns = []
    for p in pdf_paths:
        pages = PyPDFLoader(p).load()
        # Warn if all pages have empty text (likely scanned)
        if all(not (d.page_content or "").strip() for d in pages):
            warns.append(
                f"{os.path.basename(p)} has no extractable text (likely scanned). "
                "Chunk editor may look empty and boundary refine will be skipped unless OCR is used."
            )
        for d in pages:
            meta = dict(d.metadata or {})
            meta["source"] = os.path.basename(meta.get("source", p))
            # keep 'page' if present (usually 0-based)
            d.metadata = meta
        docs.extend(pages)
    return docs, warns


def split_documents(documents: List[Document], chunk_size=1000, chunk_overlap=100) -> List[Document]:
    """
    Split page-level Documents into chunks.
    Critical: add_start_index=True to remember the start offset within the page text.
    Also attach full_page_text for later boundary refinement.
    If start_index is missing (EXE/older versions), fallback by locating the substring.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks: List[Document] = []
    for page_doc in documents:
        page_text = page_doc.page_content or ""
        page_meta = dict(page_doc.metadata or {})
        parts = splitter.split_documents([page_doc])
        for d in parts:
            m = dict(d.metadata or {})
            m.setdefault("source", page_meta.get("source", "doc"))
            m.setdefault("page", page_meta.get("page"))
            m["full_page_text"] = page_text

            # Fallback: robustly infer start_index if missing
            if "start_index" not in m or m["start_index"] is None:
                t = d.page_content or ""
                pos = page_text.find(t[:200]) if t else -1
                m["start_index"] = pos if pos >= 0 else 0

            d.metadata = m
            chunks.append(d)
    return chunks


def build_vectorstore(chunks: List[Document], embeddings: OpenAIEmbeddings) -> FAISS:
    """Build FAISS VectorStore from chunks."""
    return FAISS.from_documents(chunks, embeddings)


def format_docs_for_context(docs: List[Document], max_chars=1200) -> str:
    """
    Compose a concise context: [source p.X] + excerpt. Deduplicate near-duplicates by (source,page,hash).
    """
    rows, seen = [], set()
    for d in docs:
        meta = d.metadata or {}
        name = Path(meta.get("source", "doc")).name
        page = meta.get("page")
        tag = f"[{name} p.{(page + 1) if isinstance(page, int) else '?'}]"
        text = d.page_content or ""
        key = (name, page, hash(text[:120]))
        if key in seen:
            continue
        seen.add(key)
        if len(text) > max_chars:
            text = text[:max_chars] + " ..."
        rows.append(f"{tag}\n{text}")
    return "\n\n".join(rows)


def build_chain(vectorstore: FAISS):
    """Create the base RAG generation chain (without boundary refine; refine is handled outside)."""
    # We will *not* use the retriever inside the chain because we need scores;
    # instead, we'll do retrieval ourselves and pass the refined context in.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=1)

    SYSTEM = ""  # Keep empty; you can add more strict guidance here later.
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {question}\n\nContext:\n{context}"),
        ]
    )

    rag_core = (
        RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x.get("chat_history", []),
                "context": lambda x: x["context"],
            }
        )
        | RunnableMap(
            {
                "answer": (prompt | llm | StrOutputParser()),
            }
        )
    )

    _store = {}

    def _get_history(session_id: str):
        if session_id not in _store:
            _store[session_id] = ChatMessageHistory()
        return _store[session_id]

    return RunnableWithMessageHistory(
        rag_core,
        get_session_history=_get_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )


# =========================== Retrieval & Refinement ===========================

def _cosine(u: List[float], v: List[float]) -> float:
    """Compute cosine similarity for two vectors."""
    if not u or not v:
        return 0.0
    if len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / math.sqrt(nu * nv)


def _expand_span_around(
    full_text: str, start_idx: Optional[int], end_idx: Optional[int], expand_chars: int
) -> Tuple[int, int]:
    """
    Expand [start_idx, end_idx) by +/- expand_chars within full_text bounds.
    Deterministic; no sentence-boundary heuristics.
    """
    if start_idx is None or end_idx is None:
        L = len(full_text)
        s = max(0, (L // 2) - expand_chars)
        e = min(L, s + 600 + 2 * expand_chars)
        return s, e
    s = max(0, start_idx - expand_chars)
    e = min(len(full_text), end_idx + expand_chars)
    if e < s:
        s, e = 0, min(len(full_text), 2 * expand_chars)
    return s, e


def _argmax_sliding_window(
    embeddings: OpenAIEmbeddings,
    q_vec: List[float],
    full_text: str,
    search_start: int,
    search_end: int,
    window_len: int,
    step: int,
) -> Tuple[int, int, float]:
    """
    Slide a fixed-length window (window_len) from search_start..search_end,
    with a given step size, and return (best_s, best_e, best_score).
    If search range is shorter than window_len, clamp to available region.
    """
    L = len(full_text)
    search_start = max(0, min(search_start, L))
    search_end = max(0, min(search_end, L))
    if window_len <= 0:
        return search_start, search_start, 0.0
    if search_end - search_start <= window_len:
        s0 = search_start
        e0 = min(search_start + window_len, L)
        cand = full_text[s0:e0]
        score = _cosine(q_vec, embeddings.embed_query(cand))
        return s0, e0, score

    best_s, best_e, best_score = search_start, search_start + window_len, -1.0
    s = search_start
    while s + window_len <= search_end:
        e = s + window_len
        cand = full_text[s:e]
        score = _cosine(q_vec, embeddings.embed_query(cand))
        if score > best_score:
            best_s, best_e, best_score = s, e, score
        s += max(1, step)
    return best_s, best_e, best_score


def refine_top_k_chunks(
    embeddings: OpenAIEmbeddings,
    vs: FAISS,
    query: str,
    top_k: int = 5,
    expand_chars: int = 120,
    preview_chars: int = 300,  # for right-panel preview length
    optimize_sliding: bool = True,
    sliding_step: int = 60,     # char step for sliding window
) -> Tuple[List[Document], str]:
    """
    Retrieve top-K, expand each chunk by +/- expand_chars on its page text,
    compute cosine similarity before/after, optionally run sliding-window argmax,
    and build a rich analysis string that includes file, page, score changes, spans,
    ORIGINAL and REFINED text.
    """
    from pathlib import Path

    def _shorten(txt: str, n: int) -> str:
        if not txt:
            return ""
        if len(txt) <= n:
            return txt
        return txt[:n] + " ..."

    q_vec = embeddings.embed_query(query)
    retrieved: List[Tuple[Document, float]] = vs.similarity_search_with_score(query, k=top_k)

    refined_docs: List[Document] = []
    lines: List[str] = []
    lines.append(
        f"Top-{top_k} retrieval with boundary refine (±{expand_chars} chars"
        + (", sliding" if optimize_sliding else "")
        + f"; step={sliding_step}):\n"
    )

    for rank, (doc, _faiss_score) in enumerate(retrieved, start=1):
        text = doc.page_content or ""
        meta = dict(doc.metadata or {})
        source = Path(meta.get("source", "doc")).name
        page = meta.get("page")
        page_label = (page + 1) if isinstance(page, int) else "?"
        start_index = meta.get("start_index", None)
        end_index = (start_index + len(text)) if isinstance(start_index, int) else None
        full_page_text = meta.get("full_page_text", "")

        # score before
        chunk_vec = embeddings.embed_query(text)
        score_before = _cosine(q_vec, chunk_vec)

        # initial expand
        skipped_reason = None
        if full_page_text and isinstance(start_index, int) and isinstance(end_index, int):
            s_new, e_new = _expand_span_around(full_page_text, start_index, end_index, expand_chars)
            refined_start, refined_end = s_new, e_new
            refined_text = full_page_text[refined_start:refined_end]

            # optional sliding optimization (fixed window length = original chunk length)
            if optimize_sliding:
                window_len = max(20, len(text))  # keep roughly same size as original, but >= 20
                s_min = max(0, start_index - expand_chars)
                s_max = min(len(full_page_text), end_index + expand_chars)
                best_s, best_e, best_score = _argmax_sliding_window(
                    embeddings, q_vec, full_page_text, s_min, s_max, window_len, sliding_step
                )
                # choose sliding result if it improves over the simple-expanded score
                slid_vec = embeddings.embed_query(full_page_text[best_s:best_e])
                score_slid = _cosine(q_vec, slid_vec)
                base_vec = embeddings.embed_query(refined_text)
                score_base = _cosine(q_vec, base_vec)
                if score_slid > score_base:
                    refined_start, refined_end = best_s, best_e
                    refined_text = full_page_text[refined_start:refined_end]
        else:
            refined_text = text
            refined_start, refined_end = start_index, end_index
            skipped_reason = " (refine skipped: missing start_index/full_page_text)"

        # score after
        refined_vec = embeddings.embed_query(refined_text)
        score_after = _cosine(q_vec, refined_vec)
        delta = score_after - score_before

        # build refined doc
        new_meta = dict(meta)
        new_meta.update({
            "refined": True,
            "refine_expand_chars": expand_chars,
            "orig_start_index": start_index,
            "orig_end_index": end_index,
            "refined_start_index": refined_start,
            "refined_end_index": refined_end,
            "score_before": round(score_before, 6),
            "score_after": round(score_after, 6),
            "score_delta": round(delta, 6),
        })
        refined_doc = Document(page_content=refined_text, metadata=new_meta)
        refined_docs.append(refined_doc)

        # right-panel rich block
        def _fmt_span(a, b):
            if a is None or b is None:
                return "N/A"
            return f"{a}-{b}"

        block = [
            f"#{rank} [{source} p.{page_label}]",
            f"score_before={score_before:.6f} -> score_after={score_after:.6f} (Δ={delta:+.6f})",
            f"span: { _fmt_span(start_index, end_index) } -> { _fmt_span(refined_start, refined_end) }{skipped_reason or ''}",
            "",
            "— ORIGINAL —",
            _shorten(text, preview_chars),
            "",
            "— REFINED —",
            _shorten(refined_text, preview_chars),
            "",
            "-" * 80,
            ""
        ]
        lines.extend(block)

    lines.append("Note: scores shown are cosine similarities (higher is better).")
    analysis = "\n".join(lines)
    return refined_docs, analysis


# =========================== Chunk Editor (manual stage) ===========================

class ChunkEditor(tk.Toplevel):
    """
    A modal window to preview, filter (keep/discard), and optionally edit chunk text
    before building the vector index.
    """
    def __init__(self, parent, chunks, on_done):
        super().__init__(parent)
        self.title("Review & Modify Chunks")
        self.geometry("1200x750")
        self.minsize(900, 600)
        self.parent = parent
        self.on_done = on_done

        # Each item: {"doc": Document, "keep": True}
        self.items = [{"doc": d, "keep": True} for d in chunks]

        # --- Left: list of chunk tags ---
        left = ttk.Frame(self)
        left.pack(side="left", fill="y")
        ttk.Label(left, text="Chunks").pack(anchor="w", padx=8, pady=(8, 4))

        self.listbox = tk.Listbox(left, exportselection=False, width=40)
        self.listbox.pack(fill="y", expand=False, padx=8, pady=4)
        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Populate list
        for i, it in enumerate(self.items):
            d = it["doc"]
            meta = d.metadata or {}
            name = Path(meta.get("source", "doc")).name
            page = meta.get("page")
            tag = f"[{name} p.{(page + 1) if isinstance(page, int) else '?'}] #{i+1}"
            self.listbox.insert("end", tag)

        # Batch ops
        ops = ttk.Frame(left)
        ops.pack(fill="x", padx=8, pady=6)
        ttk.Button(ops, text="Keep selected", command=self._keep_selected).pack(fill="x", pady=2)
        ttk.Button(ops, text="Discard selected", command=self._discard_selected).pack(fill="x", pady=2)
        ttk.Button(ops, text="Toggle keep/discard", command=self._toggle_selected).pack(fill="x", pady=2)

        # --- Right: editor & status ---
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True)

        # Keep/Discard status
        self.keep_var = tk.StringVar(value="Keep: True")
        ttk.Label(right, textvariable=self.keep_var).pack(anchor="w", padx=8, pady=(8, 4))

        # Text editor
        self.editor = tk.Text(right, wrap="word")
        y = ttk.Scrollbar(right, orient="vertical", command=self.editor.yview)
        self.editor["yscrollcommand"] = y.set
        self.editor.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=4)
        y.pack(side="left", fill="y", padx=(0, 8), pady=4)

        # Bottom buttons
        bottom = ttk.Frame(right)
        bottom.pack(fill="x", padx=8, pady=8)

        save_button = ttk.Button(bottom, text="Save edits", command=self._save_edits)
        save_button.pack(side="left", fill="x", expand=True, padx=(0, 4))

        done_button = ttk.Button(bottom, text="Done (build index)", command=self._done)
        done_button.pack(side="left", fill="x", expand=True, padx=(4, 0))

        # Modal-ish
        self.transient(parent)
        self.grab_set()

        # Force topmost to avoid being hidden under main window (common in EXE)
        self.lift()
        self.attributes("-topmost", True)
        self.after(300, lambda: self.attributes("-topmost", False))

        # Select first by default
        if self.items:
            self.listbox.selection_set(0)
            self._load_current(0)

    def _cur_index(self):
        sel = self.listbox.curselection()
        return sel[0] if sel else None

    def _load_current(self, idx):
        it = self.items[idx]
        self.editor.delete("1.0", "end")
        self.editor.insert("end", it["doc"].page_content or "")
        self.keep_var.set(f"Keep: {it['keep']}")

    def _on_select(self, _evt):
        idx = self._cur_index()
        if idx is not None:
            self._load_current(idx)

    def _save_edits(self):
        idx = self._cur_index()
        if idx is None:
            return
        text = self.editor.get("1.0", "end").rstrip("\n")
        self.items[idx]["doc"].page_content = text
        # If user changed text length, original end index is no longer exact,
        # but we still keep start_index and full_page_text for best-effort refine.

    def _keep_selected(self):
        for idx in self.listbox.curselection():
            self.items[idx]["keep"] = True
        self._refresh_keep_label()

    def _discard_selected(self):
        for idx in self.listbox.curselection():
            self.items[idx]["keep"] = False
        self._refresh_keep_label()

    def _toggle_selected(self):
        for idx in self.listbox.curselection():
            self.items[idx]["keep"] = not self.items[idx]["keep"]
        self._refresh_keep_label()

    def _refresh_keep_label(self):
        idx = self._cur_index()
        if idx is not None:
            self.keep_var.set(f"Keep: {self.items[idx]['keep']}")

    def _done(self):
        kept = [it["doc"] for it in self.items if it["keep"]]
        if not kept:
            if not messagebox.askyesno("No chunks kept", "You kept 0 chunks. Continue?"):
                return
        self.on_done(kept)
        self.destroy()


# =========================== Tkinter Application ===========================

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Multi-PDF Chat (All-in-one)")
        self.geometry("1100x700")

        # --- Ensure API key exists (main thread) ---
        if not os.getenv("OPENAI_API_KEY"):
            self.withdraw()
            try:
                key = tk.simpledialog.askstring("API Key", "Enter OPENAI_API_KEY:")
                if key:
                    os.environ["OPENAI_API_KEY"] = key.strip()
                else:
                    messagebox.showerror("Error", "Missing OPENAI_API_KEY. Exiting.")
                    self.destroy()
                    raise SystemExit(1)
            finally:
                self.deiconify()

        # Top toolbar
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x")

        self.btn_build = ttk.Button(bar, text="Select PDFs & Build Index", command=self.on_build)
        self.btn_build.pack(side="left", padx=8, pady=6)

        # Add: reopen editor anytime
        self._last_chunks: List[Document] = []
        ttk.Button(bar, text="Reopen Chunk Editor",
                   command=lambda: self._open_chunk_editor(self._last_chunks or [])).pack(side="left", padx=4, pady=6)

        # Retrieval/Refine toolbar controls
        ttk.Label(bar, text="Top-K:").pack(side="left", padx=(16, 4))
        self.topk_var = tk.IntVar(value=5)
        ttk.Spinbox(bar, from_=1, to=20, textvariable=self.topk_var, width=4).pack(side="left")

        ttk.Label(bar, text="Refine ±chars:").pack(side="left", padx=(12, 4))
        self.expand_var = tk.IntVar(value=120)
        ttk.Spinbox(bar, from_=0, to=1000, textvariable=self.expand_var, width=5).pack(side="left")

        self.enable_refine = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Enable refine", variable=self.enable_refine).pack(side="left", padx=(12, 0))

        self.enable_sliding = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Optimize (sliding)", variable=self.enable_sliding).pack(side="left", padx=(8, 0))

        ttk.Label(bar, text="Step:").pack(side="left", padx=(12, 4))
        self.step_var = tk.IntVar(value=60)
        ttk.Spinbox(bar, from_=10, to=200, textvariable=self.step_var, width=5).pack(side="left")

        ttk.Label(bar, text="Preview chars:").pack(side="left", padx=(12, 4))
        self.preview_chars_var = tk.IntVar(value=300)
        ttk.Spinbox(bar, from_=80, to=2000, textvariable=self.preview_chars_var, width=5).pack(side="left")

        self.status = tk.StringVar(value="Status: idle")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=12)

        # Main panes: left chat, right analysis
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        # Left (chat)
        left = ttk.Frame(panes)
        self.chat = tk.Text(left, wrap="word")
        self.chat.config(state="disabled")
        y1 = ttk.Scrollbar(left, orient="vertical", command=self.chat.yview)
        self.chat["yscrollcommand"] = y1.set
        self.chat.pack(side="left", fill="both", expand=True)
        y1.pack(side="right", fill="y")
        panes.add(left, weight=3)

        # Right (analysis)
        right = ttk.Frame(panes)
        self.analysis = tk.Text(right, wrap="word", foreground="#444")
        self.analysis.config(state="disabled")
        y2 = ttk.Scrollbar(right, orient="vertical", command=self.analysis.yview)
        self.analysis["yscrollcommand"] = y2.set
        self.analysis.pack(side="left", fill="both", expand=True)
        y2.pack(side="right", fill="y")
        panes.add(right, weight=2)

        # Bottom input
        bottom = ttk.Frame(self)
        bottom.pack(side="bottom", fill="x", padx=8, pady=8)

        self.entry = ttk.Entry(bottom)
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", self.on_send)

        self.btn_send = ttk.Button(bottom, text="Send", command=self.on_send, state="disabled")
        self.btn_send.pack(side="left", padx=8)

        # State
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.qa = None              # generation chain (no retriever inside)
        self.vectorstore = None     # FAISS
        self.session_id = "default_session"
        self.q = queue.Queue()
        self._editor = None         # hold reference to ChunkEditor

        # Background UI polling
        self.after(100, self._poll)

        # Windows perf quirks
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ------------------ UI helpers ------------------

    def log_chat(self, who: str, msg: str):
        self.chat.config(state="normal")
        self.chat.insert("end", f"{who}: {msg}\n\n")
        self.chat.see("end")
        self.chat.config(state="disabled")

    def log_analysis(self, text: str):
        self.analysis.config(state="normal")
        self.analysis.delete("1.0", "end")
        self.analysis.insert("end", text)
        self.analysis.see("end")
        self.analysis.config(state="disabled")

    # ------------------ Build pipeline ------------------

    def on_build(self):
        """Pick PDFs, split pages -> chunks, open manual editor, then embed & index."""
        paths = pick_pdfs(self)
        if not paths:
            return
        self.btn_build.config(state="disabled")
        self.btn_send.config(state="disabled")
        self.status.set("Status: loading PDFs...")
        self.log_chat("System", f"Selected {len(paths)} file(s). Preparing chunks...")

        t = threading.Thread(target=self._worker_prepare, args=(paths,), daemon=True)
        t.start()

    def _worker_prepare(self, paths: List[str]):
        try:
            self.status.set("Status: loading documents")
            documents, warns = load_documents(paths)

            # Push warnings to main thread to show later (no messagebox here!)
            for w in warns:
                self.q.put(("alert", "warning", "No extractable text", w))

            self.status.set("Status: splitting")
            chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)

            # Cache for reopen
            self._last_chunks = chunks

            # hand off to main thread to open editor
            self.q.put(("open_editor", chunks))
            self.status.set("Status: review & modify chunks")
            self.q.put(("chat", "System", f"Chunks prepared: {len(chunks)}. Review & modify before indexing."))
        except Exception as e:
            self.q.put(("chat", "Error", f"Prepare failed: {e}"))
            self.q.put(("enable_build",))
            self.status.set("Status: idle")

    def _worker_embed_index(self, chunks: List[Document]):
        try:
            self.status.set("Status: embedding & indexing")
            self.vectorstore = build_vectorstore(chunks, self.embeddings)

            self.status.set("Status: composing chain")
            self.qa = build_chain(self.vectorstore)

            self.q.put(("chat", "System", f"Index ready. Kept chunks: {len(chunks)}. You can chat now."))
            self.q.put(("enable",))
            self.status.set("Status: ready")
        except Exception as e:
            self.q.put(("chat", "Error", f"Build failed: {e}"))
            self.q.put(("enable_build",))
            self.status.set("Status: idle")

    def _open_chunk_editor(self, chunks: List[Document]):
        """Open the manual chunk editor. After user clicks Done, proceed to index."""
        if not chunks:
            messagebox.showinfo("Info", "No chunks to review. Did your PDF contain extractable text?")
            return

        def _on_done(kept_chunks):
            threading.Thread(target=self._worker_embed_index, args=(kept_chunks,), daemon=True).start()

        # Hold a reference to avoid GC; force topmost to avoid being hidden in EXE
        self._editor = ChunkEditor(self, chunks, on_done=_on_done)
        self._editor.lift()
        self._editor.attributes("-topmost", True)
        self._editor.after(300, lambda: self._editor.attributes("-topmost", False))

    # ------------------ Chat flow ------------------

    def _compose_context_from_docs(self, docs: List[Document]) -> str:
        """Compose final context string for LLM."""
        return format_docs_for_context(docs, max_chars=1200)

    def on_send(self, event=None):
        text = self.entry.get().strip()
        if text == "":  # blank Enter -> exit
            self.destroy()
            return
        if not self.qa or not self.vectorstore:
            messagebox.showinfo("Info", "Please build the index first.")
            return
        self.entry.delete(0, "end")
        self.log_chat("You", text)
        threading.Thread(target=self._worker_ask, args=(text,), daemon=True).start()

    def _worker_ask(self, question: str):
        try:
            # Retrieve & (optionally) refine
            top_k = max(1, int(self.topk_var.get() or 5))
            expand_chars = max(0, int(self.expand_var.get() or 120))
            preview_chars = max(80, int(self.preview_chars_var.get() or 300))
            optimize_sliding = bool(self.enable_sliding.get())
            step = max(5, int(self.step_var.get() or 60))

            if self.enable_refine.get():
                refined_docs, analysis = refine_top_k_chunks(
                    embeddings=self.embeddings,
                    vs=self.vectorstore,
                    query=question,
                    top_k=top_k,
                    expand_chars=expand_chars,
                    preview_chars=preview_chars,
                    optimize_sliding=optimize_sliding,
                    sliding_step=step,
                )
                context = self._compose_context_from_docs(refined_docs)
                analysis_text = f"{analysis}\n\nContext uses refined snippets."
            else:
                # No refine: just use similarity_search
                raw_docs = [d for (d, _s) in self.vectorstore.similarity_search_with_score(question, k=top_k)]
                context = self._compose_context_from_docs(raw_docs)
                from pathlib import Path as _Path
                names = []
                for i, d in enumerate(raw_docs, start=1):
                    m = d.metadata or {}
                    names.append(f"#{i} [{_Path(m.get('source','doc')).name} p.{(m.get('page')+1) if isinstance(m.get('page'),int) else '?'}]")
                analysis_text = "Top-K retrieval (no refine):\n" + "\n".join(names)

            # Run generation
            res = self.qa.invoke(
                {"question": question, "context": context},
                config={"configurable": {"session_id": self.session_id}},
            )
            answer = res.get("answer", "")

            # Right panel text
            self.q.put(("chat", "Bot", answer))
            self.q.put(("analysis", analysis_text if analysis_text else "(No analysis)"))

        except Exception as e:
            self.q.put(("chat", "Error", str(e)))

    # ------------------ Thread-safe UI updates ------------------

    def _poll(self):
        try:
            while True:
                item = self.q.get_nowait()
                tag = item[0]
                if tag == "chat":
                    _, who, msg = item
                    self.log_chat(who, msg)
                elif tag == "analysis":
                    _, text = item
                    self.log_analysis(text)
                elif tag == "enable":
                    self.btn_build.config(state="normal")
                    self.btn_send.config(state="normal")
                elif tag == "enable_build":
                    self.btn_build.config(state="normal")
                elif tag == "open_editor":
                    _, chunks = item
                    self._open_chunk_editor(chunks)
                elif tag == "alert":
                    # ("alert", level, title, message)
                    _, level, title, msg = item
                    if level == "warning":
                        messagebox.showwarning(title, msg)
                    elif level == "error":
                        messagebox.showerror(title, msg)
                    else:
                        messagebox.showinfo(title, msg)
        except queue.Empty:
            pass
        self.after(100, self._poll)


# =========================== Main ===========================

if __name__ == "__main__":
    app = App()
    app.mainloop()
