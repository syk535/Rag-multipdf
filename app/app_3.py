# app_3.py
# Minimal RAG GUI that shows Sources + Retrieved snippets on the right panel


import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure OpenAI encodings (e.g., cl100k_base) are registered in frozen builds
import tiktoken_ext.openai_public  # noqa: F401

# ---------------------- PyInstaller: pin Tcl/Tk paths when frozen ----------------------
def _pin_tcl_tk_for_pyinstaller():
    """
    When frozen, point Tcl/Tk to the copies that PyInstaller bundles.
    """
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

# ---------------------- API key loading strategy ----------------------
# Prefer .env next to the exe/script; fallback to CWD
dotenv_file = APP_PATH / ".env"
if dotenv_file.exists():
    load_dotenv(dotenv_file)
else:
    load_dotenv()

# Fallback: read OPENAI_API_KEY from plaintext file next to exe/script
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

# ---------------------- Standard libs ----------------------
import threading
import queue
from typing import List

# ---------------------- GUI ----------------------
import tkinter as tk
from tkinter import filedialog, ttk, messagebox

# ---------------------- LangChain / OpenAI stack ----------------------
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# ---------------------- RAG helpers ----------------------
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
    """
    Format retrieved docs into a compact context block.
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
    """
    Build a simple RAG chain.
    Retriever: MMR (diversity) with k=5 from fetch_k=30
    LLM: gpt-4o-mini
    Output: answer + (we also pass-through docs/snippets/sources for the UI)
    """
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
            # Pass-through for UI (right panel)
            "docs":     lambda x: x["docs"],
            "snippets": lambda x: [d.page_content for d in x["docs"]],
            "sources":  lambda x: [
                {
                    "source": Path((d.metadata or {}).get("source", "doc")).name,
                    "page":   ((d.metadata or {}).get("page") + 1)
                             if isinstance((d.metadata or {}).get("page"), int) else None
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


# ---------------------- Tkinter GUI ----------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RAG Multi-PDF Chat (Sources + Snippets)")
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

        # Poll queue to update UI from worker threads
        self.after(100, self._poll)

        # Windows perf quirks
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ---------------------- Logging helpers ----------------------
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

    # ---------------------- Build pipeline ----------------------
    def on_build(self):
        paths = pick_pdfs(self)
        if not paths:
            return
        self.btn_build.config(state="disabled")
        self.btn_send.config(state="disabled")
        self.status.set("Status: loading PDFs...")
        self.log_chat("System", f"Selected {len(paths)} file(s). Preparing chunks...")

        t = threading.Thread(target=self._worker_build, args=(paths,), daemon=True)
        t.start()

    def _worker_build(self, paths: List[str]):
        try:
            self.status.set("Status: loading documents")
            documents = load_documents(paths)

            self.status.set("Status: splitting")
            chunks = split_documents(documents, chunk_size=1000, chunk_overlap=100)

            self.status.set("Status: embedding & indexing")
            self.vectorstore = build_vectorstore(chunks, self.embeddings)

            self.status.set("Status: composing chain")
            self.qa = build_chain(self.vectorstore)

            self.q.put(("chat", "System", f"Index ready. Chunks: {len(chunks)}. You can chat now."))
            self.q.put(("enable",))
            self.status.set("Status: ready")
        except Exception as e:
            self.q.put(("chat", "Error", f"Build failed: {e}"))
            self.q.put(("enable_build",))
            self.status.set("Status: idle")

    # ---------------------- Chat ----------------------
    def on_send(self, event=None):
        text = self.entry.get().strip()
        if text == "":
            self.destroy()  # blank Enter closes app (same behavior as earlier)
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

            # Compose right panel: sources + first few snippets (trimmed)
            right_text = ""
            if sources:
                right_text += "Sources:\n" + "\n".join(
                    f"  - {s['source']}" + (f" p.{s['page']}" if s.get('page') else "")
                    for s in sources
                ) + "\n\n"
            if snippets:
                right_text += "Retrieved snippets:\n" + "\n\n".join(
                    (snip if len(snip) <= 600 else (snip[:600] + " ..."))
                    for snip in snippets[:5]   # show at most 5
                )

            self.q.put(("chat", "Bot", answer))
            self.q.put(("analysis", right_text if right_text else "(No snippets/sources)"))
        except Exception as e:
            self.q.put(("chat", "Error", str(e)))

    # ---------------------- Thread-safe UI updates ----------------------
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
        except queue.Empty:
            pass
        self.after(100, self._poll)


if __name__ == "__main__":
    app = App()
    app.mainloop()
