# --- EXE-compatible path & API key loader ---
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

import tiktoken_ext.openai_public  # make sure OpenAI encodings (e.g., cl100k_base) are registered

# --- Point Tcl/Tk to the bundled copy when frozen (PyInstaller) ---
def _pin_tcl_tk_for_pyinstaller():
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base and os.path.isdir(base):
            tcl = os.path.join(base, "tcl", "tcl8.6")
            tk  = os.path.join(base, "tcl", "tk8.6")
            os.environ.setdefault("TCL_LIBRARY", tcl)
            os.environ.setdefault("TK_LIBRARY",  tk)
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

# Final fallback: prompt once via Tk (optional). Replace with raise if you prefer.
if not os.getenv("OPENAI_API_KEY"):
    try:
        import tkinter as tk
        from tkinter import simpledialog, messagebox
        root = tk.Tk(); root.withdraw()
        key = simpledialog.askstring("API Key", "Enter OPENAI_API_KEY:")
        if key:
            os.environ["OPENAI_API_KEY"] = key.strip()
            messagebox.showinfo("OK", "API key captured for this run.")
        else:
            raise RuntimeError("Missing OPENAI_API_KEY")
    except Exception:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Place it in a .env file or OPENAI_API_KEY.txt "
            "next to the executable (or this script)."
        )

import threading
import queue
from typing import List

# GUI
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# LangChain / OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ---------- RAG helpers ----------
def pick_pdfs(parent) -> List[str]:
    return list(filedialog.askopenfilenames(
        parent=parent, title="Select PDF files", filetypes=[("PDF files", "*.pdf")]
    ))

def load_documents(pdf_paths: List[str]):
    docs = []
    for p in pdf_paths:
        pages = PyPDFLoader(p).load()
        for d in pages:
            meta = d.metadata or {}
            meta["source"] = os.path.basename(meta.get("source", p))
            d.metadata = meta
        docs.extend(pages)
    return docs

def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)

def build_vectorstore(chunks, embeddings) -> FAISS:
    return FAISS.from_documents(chunks, embeddings)

def format_docs_for_context(docs, max_chars=1200):
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
    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 30, "lambda_mult": 0.5}
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=1)

    SYSTEM = ""  # keep empty for now; you can add stricter rules later
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])

    rag_core = (
        RunnableMap({
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        })
        | RunnableMap({
            "docs":        lambda x: retriever.invoke(x["question"]),
            "question":    lambda x: x["question"],
            "chat_history":lambda x: x["chat_history"],
        })
        | RunnableMap({
            "context":      lambda x: format_docs_for_context(x["docs"]),
            "question":     lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
            "docs":         lambda x: x["docs"],
        })
        | RunnableMap({
            "answer":   (prompt | llm | StrOutputParser()),
            "docs":     lambda x: x["docs"],
            "snippets": lambda x: [d.page_content for d in x["docs"]],
            "sources":  lambda x: [
                {
                    "source": Path((d.metadata or {}).get("source", "doc")).name,
                    "page":   ((d.metadata or {}).get("page") + 1) if isinstance((d.metadata or {}).get("page"), int) else None
                } for d in x["docs"]
            ],
        })
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
        output_messages_key="answer"
    )


# ---------- Chunk Editor (preview/filter/edit) ----------
class ChunkEditor(tk.Toplevel):
    """
    A modal window to preview, filter (keep/discard), and optionally edit chunk text
    before building the vector index.
    """
    def __init__(self, parent, chunks, on_done):
        super().__init__(parent)
        self.title("Review & Modify Chunks")
        self.geometry("1000x600")
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
        ttk.Button(bottom, text="Save edits", command=self._save_edits).pack(side="left")
        ttk.Button(bottom, text="Done (build index)", command=self._done).pack(side="right")

        # Modal-ish
        self.transient(parent)
        self.grab_set()

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


# ---------- Tkinter GUI ----------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Multi-PDF Chat (All-in-one)")
        self.geometry("1000x640")

        # Top toolbar
        bar = ttk.Frame(self)
        bar.pack(side="top", fill="x")
        self.btn_build = ttk.Button(bar, text="Select PDFs & Build Index", command=self.on_build)
        self.btn_build.pack(side="left", padx=8, pady=6)
        self.status = tk.StringVar(value="Status: idle")
        ttk.Label(bar, textvariable=self.status).pack(side="left", padx=10)

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
        self.qa = None
        self.vectorstore = None
        self.session_id = "default_session"
        self.q = queue.Queue()

        self.after(100, self._poll)

        # Windows perf quirks
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

    # ---- Build pipeline in two stages: prepare -> (edit) -> index ----
    def on_build(self):
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
            documents = load_documents(paths)

            self.status.set("Status: splitting")
            chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)

            # hand off to main thread to open editor
            self.q.put(("open_editor", chunks))
            self.status.set("Status: review & modify chunks")
            self.q.put(("chat", "System", f"Chunks prepared: {len(chunks)}. Review & modify before indexing."))
        except Exception as e:
            self.q.put(("chat", "Error", f"Prepare failed: {e}"))
            self.q.put(("enable_build",))
            self.status.set("Status: idle")

    def _worker_embed_index(self, chunks):
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

    def _open_chunk_editor(self, chunks):
        def _on_done(kept_chunks):
            # Start indexing after user confirms edits
            threading.Thread(target=self._worker_embed_index, args=(kept_chunks,), daemon=True).start()
        ChunkEditor(self, chunks, on_done=_on_done)

    # ---- Chat ----
    def on_send(self, event=None):
        text = self.entry.get().strip()
        if text == "":           # blank Enter -> exit
            self.destroy()
            return
        if not self.qa:
            messagebox.showinfo("Info", "Please build the index first.")
            return
        self.entry.delete(0, "end")
        self.log_chat("You", text)
        threading.Thread(target=self._worker_ask, args=(text,), daemon=True).start()

    def _worker_ask(self, question: str):
        try:
            res = self.qa.invoke(
                {"question": question},
                config={"configurable": {"session_id": self.session_id}}
            )
            answer = res.get("answer", "")
            sources = res.get("sources", [])
            snippets = res.get("snippets", [])

            # Right panel text
            right_text = ""
            if sources:
                right_text += "Sources:\n" + "\n".join(
                    f"  - {s['source']}" + (f" p.{s['page']}" if s.get('page') else "")
                    for s in sources
                ) + "\n\n"
            if snippets:
                right_text += "Retrieved snippets:\n" + "\n\n".join(
                    (snip if len(snip) <= 600 else (snip[:600] + " ..."))
                    for snip in snippets[:5]
                )

            self.q.put(("chat", "Bot", answer))
            self.q.put(("analysis", right_text if right_text else "(No snippets/sources)"))
        except Exception as e:
            self.q.put(("chat", "Error", str(e)))

    # ---- Thread-safe UI updates ----
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
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    app = App()
    app.mainloop()
