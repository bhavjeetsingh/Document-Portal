from __future__ import annotations
import os
import sys
import json
import uuid
import hashlib
import shutil
import pandas as pd
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any, Union
import fitz  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from utils.file_io import generate_session_id, save_uploaded_files
from utils.document_ops import load_documents, concat_for_analysis, concat_for_comparison

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt",".md", ".ppt", ".pptx", 
    ".xlsx", ".xls", ".csv", ".json", ".rtf"}

# for database connections (for sql support)
SUPPORTED_DB_TYPES={
    'postgresql','mysql','sqlite','mongodb','oracle'
}

# FAISS Manager (load-or-create)
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}} ## this is dict of rows
        
        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}} # load it if alrady there
            except Exception:
                self._meta = {"rows": {}} # init the empty one if dones not exists
        

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None
        
    def _exists(self)-> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()
    
    @staticmethod
    def _fingerprint(text: str, md: Dict[str, Any]) -> str:
        src = md.get("source") or md.get("file_path")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    
    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, ensure_ascii=False, indent=2), encoding="utf-8")
        
        
    def add_documents(self,docs: List[Document]):
        
        if self.vs is None:
            raise RuntimeError("Call load_or_create() before add_documents_idempotent().")
        
        new_docs: List[Document] = []
        
        for d in docs:
            
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key in self._meta["rows"]:
                continue
            self._meta["rows"][key] = True
            new_docs.append(d)
            
        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)
    
    def load_or_create(self,texts:Optional[List[str]]=None, metadatas: Optional[List[dict]] = None):
        ## if we running first time then it will not go in this block
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        
        
        if not texts:
            raise DocumentPortalException("No existing FAISS index and no data to create one", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs
        
        
class ChatIngestor:
    def __init__( self,
        temp_base: str = "data",
        faiss_base: str = "faiss_index",
        use_session_dirs: bool = True,
        session_id: Optional[str] = None,
    ):
        try:
            self.model_loader = ModelLoader()
            
            self.use_session = use_session_dirs
            self.session_id = session_id or generate_session_id()
            
            self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
            self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)
            
            self.temp_dir = self._resolve_dir(self.temp_base)
            self.faiss_dir = self._resolve_dir(self.faiss_base)

            log.info("ChatIngestor initialized",
                      session_id=self.session_id,
                      temp_dir=str(self.temp_dir),
                      faiss_dir=str(self.faiss_dir),
                      sessionized=self.use_session)
        except Exception as e:
            log.error("Failed to initialize ChatIngestor", error=str(e))
            raise DocumentPortalException("Initialization error in ChatIngestor", e) from e
            
        
    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id # e.g. "faiss_index/abc123"
            d.mkdir(parents=True, exist_ok=True) # creates dir if not exists
            return d
        return base # fallback: "faiss_index/"
        
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info("Documents split", chunks=len(chunks), chunk_size=chunk_size, overlap=chunk_overlap)
        return chunks
    
    def built_retriver( self,
        uploaded_files: Iterable,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")
            
            chunks = self._split(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            
            ## FAISS manager very very important class for the docchat
            fm = FaissManager(self.faiss_dir, self.model_loader)
            
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]
            
            try:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
            except Exception:
                vs = fm.load_or_create(texts=texts, metadatas=metas)
                
            added = fm.add_documents(chunks)
            log.info("FAISS index updated", added=added, index=str(self.faiss_dir))
            
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})
            
        except Exception as e:
            log.error("Failed to build retriever", error=str(e))
            raise DocumentPortalException("Failed to build retriever", e) from e

            
        
            
class DocHandler:
    """
    Universal document save + read fir analysis - supports all document types.
    """
    def __init__(self, data_dir: Optional[str] = None, session_id: Optional[str] = None):
        self.data_dir = data_dir or os.getenv("DATA_STORAGE_PATH", os.path.join(os.getcwd(), "data", "document_analysis"))
        self.session_id = session_id or generate_session_id("session")
        self.session_path = os.path.join(self.data_dir, self.session_id)
        os.makedirs(self.session_path, exist_ok=True)
        log.info("DocHandler initialized", session_id=self.session_id, session_path=self.session_path)

    def save_document(self, uploaded_file) -> str:
        # updated save any supported document type
        try:
            filename = os.path.basename(uploaded_file.name)
            file_ext = Path(filename).suffix.lower()
            
            if file_ext not in SUPPORTED_EXTENSIONS:
                raise ValueError(f'Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}')
            save_path = os.path.join(self.session_path, filename)
            with open(save_path, "wb") as f:
                if hasattr(uploaded_file, "read"):
                    f.write(uploaded_file.read())
                else:
                    f.write(uploaded_file.getbuffer())
                    
            log.info("Document saved successfully", file=filename, type=file_ext, save_path=save_path, session_id=self.session_id)
            return save_path
        except Exception as e:
            log.error("Failed to save document", error=str(e), session_id=self.session_id)
            raise DocumentPortalException(f"Failed to save document: {str(e)}", e) from e

    def read_document(self, doc_path: str) -> str:
        # now def is updated  before it was read_pdf
        try:
            file_path = Path(doc_path)
            file_ext = file_path.suffix.lower()
            
            if file_ext == '.pdf':
                return self._read_pdf(doc_path)
            elif file_ext in ['.docx']:
                return self._read_docx(doc_path)
            elif file_ext in ['.ppt', '.pptx']:
                return self._read_ppt(doc_path)
            elif file_ext in ['.xlsx', '.xls']:
                return self._read_excel(doc_path)
            elif file_ext == '.csv':
                return self._read_csv(doc_path)
            elif file_ext in ['.txt','.md']:
                return self._read_text(doc_path)
            elif file_ext == '.json':
                return self._read_json(doc_path)
        except Exception as e:
            log.error('Failed to read document', error=str(e), doc_path=doc_path, session_id=self.session_id)
            raise DocumentPortalException(f'Could not process document: {doc_path}', e) from e
    
    
    def _read_pdf(self, pdf_path: str) -> str:
        """Original PDF reading logic"""
        text_chunks = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                text_chunks.append(f"\n--- Page {page_num + 1} ---\n{page.get_text()}")
        text = "\n".join(text_chunks)
        log.info("PDF read successfully", pdf_path=pdf_path, pages=len(text_chunks))
        return text

    def _read_docx(self, docx_path: str) -> str:
        """Read Word documents - placeholder for implementation"""
        # Will implement with python-docx
        log.info("DOCX reading - placeholder", path=docx_path)
        return f"DOCX content from {docx_path} - TO BE IMPLEMENTED"

    def _read_ppt(self, ppt_path: str) -> str:
        """Read PowerPoint documents - placeholder for implementation"""
        # Will implement with python-pptx
        log.info("PPT reading - placeholder", path=ppt_path)
        return f"PPT content from {ppt_path} - TO BE IMPLEMENTED"

    def _read_excel(self, excel_path: str) -> str:
        """Read Excel documents - placeholder for implementation"""
        # Will implement with pandas/openpyxl
        log.info("Excel reading - placeholder", path=excel_path)
        return f"Excel content from {excel_path} - TO BE IMPLEMENTED"

    def _read_csv(self, csv_path: str) -> str:
        """Read CSV documents - placeholder for implementation"""
        # Will implement with pandas
        log.info("CSV reading - placeholder", path=csv_path)
        return f"CSV content from {csv_path} - TO BE IMPLEMENTED"

    def _read_text(self, text_path: str) -> str:
        """Read plain text and markdown files"""
        with open(text_path, 'r', encoding='utf-8') as f:
            content = f.read()
        log.info("Text file read successfully", path=text_path)
        return content

    def _read_json(self, json_path: str) -> str:
        """Read JSON files"""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        log.info("JSON file read successfully", path=json_path)
        return content    
        
        
class DocumentComparator:
    """
    Save, read & combine any document type for comparison with session-based versioning.
    """
    def __init__(self, base_dir: str = "data/document_compare", session_id: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.session_id = session_id or generate_session_id()
        self.session_path = self.base_dir / self.session_id
        self.session_path.mkdir(parents=True, exist_ok=True)
        log.info("DocumentComparator initialized", session_path=str(self.session_path))

    def save_uploaded_files(self, reference_file, actual_file):
        """UPDATED: Save any supported document types (was PDF only)"""
        try:
            ref_path = self.session_path / reference_file.name
            act_path = self.session_path / actual_file.name
            
            # Validate file types
            for file_obj, path in ((reference_file, ref_path), (actual_file, act_path)):
                file_ext = path.suffix.lower()
                if file_ext not in SUPPORTED_EXTENSIONS:
                    raise ValueError(f"Unsupported file type: {file_ext}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
            
            # Save files
            for fobj, out in ((reference_file, ref_path), (actual_file, act_path)):
                with open(out, "wb") as f:
                    if hasattr(fobj, "read"):
                        f.write(fobj.read())
                    else:
                        f.write(fobj.getbuffer())
                        
            log.info("Files saved", reference=str(ref_path), actual=str(act_path), session=self.session_id)
            return ref_path, act_path
            
        except Exception as e:
            log.error("Error saving files", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error saving files", e) from e
    def read_document(self, doc_path: Path) -> str:
        """UPDATED: Read any document type (was read_pdf only)"""
        try:
            # Use DocHandler for universal document reading
            doc_handler = DocHandler()
            content = doc_handler.read_document(str(doc_path))
            log.info("Document read successfully", file=str(doc_path))
            return content
            
        except Exception as e:
            log.error("Error reading document", file=str(doc_path), error=str(e))
            raise DocumentPortalException("Error reading document", e) from e

    def combine_documents(self) -> str:
        """UPDATED: Combine any document types (was PDF only)"""
        try:
            doc_parts = []
            for file in sorted(self.session_path.iterdir()):
                if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    content = self.read_document(file)
                    doc_parts.append(f"Document: {file.name}\n{content}")
            
            combined_text = "\n\n".join(doc_parts)
            log.info("Documents combined", count=len(doc_parts), session=self.session_id)
            return combined_text
            
        except Exception as e:
            log.error("Error combining documents", error=str(e), session=self.session_id)
            raise DocumentPortalException("Error combining documents", e) from e

    def clean_old_sessions(self, keep_latest: int = 3):
        """Unchanged - cleanup old sessions"""
        try:
            sessions = sorted([f for f in self.base_dir.iterdir() if f.is_dir()], reverse=True)
            for folder in sessions[keep_latest:]:
                shutil.rmtree(folder, ignore_errors=True)
                log.info("Old session folder deleted", path=str(folder))
        except Exception as e:
            log.error("Error cleaning old sessions", error=str(e))
            raise DocumentPortalException("Error cleaning old sessions", e) from e

