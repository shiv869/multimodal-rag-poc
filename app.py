import os
import base64
import lancedb
import gradio as gr
from typing import TypedDict, List, Dict, Any

# Colab Secret Retrieval
from google.colab import userdata

# Document Parsing (Docling)
from docling.document_converter import DocumentConverter

# LangChain & Google Gemini Integrations
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import LanceDB
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Reranking & Orchestration
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph, END

# =====================================================================
# SAFE SECRETS RETRIEVAL (NO HARDCODED KEYS)
# =====================================================================
try:
    GEMINI_KEY = userdata.get('GEMINI_API_KEY')
except Exception:
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise ValueError("GEMINI_API_KEY secret not found in Colab Secrets (Key icon 🔑)!")

# =====================================================================
# STEP 1: LAYOUT-AWARE INGESTION AGENT (DOCLING)
# =====================================================================
class LayoutAwareIngestionAgent:
    """Parses PDF, Word, and PPTX documents into structured Markdown using Docling."""
    
    def __init__(self):
        self.converter = DocumentConverter()

    def process_file(self, file_path: str) -> Dict[str, Any]:
        ext = os.path.splitext(file_path)[1].lower().replace(".", "")
        file_name = os.path.basename(file_path)
        source_id = f"{ext}_{file_name}"
        
        conversion_result = self.converter.convert(file_path)
        doc = conversion_result.document
        markdown_text = doc.export_to_markdown()

        return {
            "source_id": source_id,
            "file_path": file_path,
            "source_type": ext,
            "markdown_content": markdown_text
        }

# =====================================================================
# STEP 2: SPECIALIZED CHUNKER (GEMINI EMBEDDINGS + VLM)
# =====================================================================
class GeminiSpecializedChunker:
    """Applies layout-aware + semantic chunking and VLM diagram captioning using Gemini."""
    
    def __init__(self, gemini_api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=gemini_api_key
        )
        self.vlm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            google_api_key=gemini_api_key,
            temperature=0
        )
        
        self.headers_to_split = [("#", "Chapter"), ("##", "Section"), ("###", "SubSection")]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split, 
            strip_headers=False
        )
        self.semantic_splitter = SemanticChunker(
            self.embeddings, 
            breakpoint_threshold_type="percentile"
        )

    def chunk_prose(self, markdown_text: str, base_metadata: dict) -> List[Document]:
        header_splits = self.header_splitter.split_text(markdown_text)
        final_docs = []
        
        for split in header_splits:
            chunk_metadata = {**base_metadata, **split.metadata, "element_type": "prose"}
            
            if len(split.page_content) < 400:
                final_docs.append(Document(page_content=split.page_content, metadata=chunk_metadata))
            else:
                semantic_chunks = self.semantic_splitter.create_documents(
                    [split.page_content], 
                    metadatas=[chunk_metadata]
                )
                final_docs.extend(semantic_chunks)
            
        return final_docs

    def caption_image_or_diagram(self, image_path: str, base_metadata: dict) -> Document:
        with open(image_path, "rb") as img_file:
            base64_img = base64.b64encode(img_file.read()).decode("utf-8")

        msg = HumanMessage(
            content=[
                {"type": "text", "text": "Describe this flowchart/diagram in complete technical detail, including all decision nodes, arrows, and labels."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_img}"}}
            ]
        )
        vlm_response = self.vlm.invoke([msg])
        
        meta = {**base_metadata, "element_type": "flowchart", "image_path": image_path}
        return Document(
            page_content=f"[FLOWCHART/DIAGRAM DESCRIPTION]: {vlm_response.content}", 
            metadata=meta
        )

# =====================================================================
# STEP 3: LANCEDB VECTOR STORE & FULL-TEXT SEARCH INDEXING
# =====================================================================
class LanceDBManager:
    """Manages document embedding, storage, and native FTS indexing in LanceDB."""
    
    def __init__(self, table_name: str, db_uri: str, embedding_function):
        self.table_name = table_name
        self.db_uri = db_uri
        self.embedding_function = embedding_function
        self.db = lancedb.connect(db_uri)
        self.vector_store = None

    def add_chunks(self, docs: List[Document]):
        self.vector_store = LanceDB.from_documents(
            documents=docs,
            embedding=self.embedding_function,
            connection=self.db,
            table_name=self.table_name,
            mode="overwrite"
        )
        table = self.db.open_table(self.table_name)
        table.create_fts_index("text", replace=True)

    def get_table(self):
        return self.db.open_table(self.table_name)

# =====================================================================
# STEP 4: HYBRID RETRIEVAL + CROSS-ENCODER RERANKING
# =====================================================================
class LanceDBHybridRerankRetriever:
    """Performs LanceDB Hybrid Search (Dense + Sparse) followed by Neural Reranking."""
    
    def __init__(self, lancedb_manager: LanceDBManager, reranker_model: str = "bge-reranker-base"):
        self.db_manager = lancedb_manager
        self.reranker = CrossEncoder(reranker_model)

    def retrieve(self, query: str, top_k: int = 5, fetch_k: int = 20) -> List[Document]:
        table = self.db_manager.get_table()
        
        hybrid_df = (
            table.search(query, query_type="hybrid")
            .limit(fetch_k)
            .to_pandas()
        )
        
        candidates = []
        for _, row in hybrid_df.iterrows():
            metadata = {k: v for k, v in row.to_dict().items() if k not in ["text", "vector", "_distance", "_relevance_score"]}
            candidates.append(Document(page_content=row["text"], metadata=metadata))

        if not candidates:
            return []

        pairs = [[query, doc.page_content] for doc in candidates]
        rerank_scores = self.reranker.predict(pairs)

        scored_pairs = sorted(zip(candidates, rerank_scores), key=lambda x: x, reverse=True)
        return [doc for doc, score in scored_pairs[:top_k]]

# =====================================================================
# STEP 5: LANGGRAPH RAG PIPELINE
# =====================================================================
class RAGState(TypedDict):
    query: str
    context_docs: List[Document]
    answer: str

class LangGraphLanceDBPipeline:
    def __init__(self, retriever: LanceDBHybridRerankRetriever, gemini_api_key: str):
        self.retriever = retriever
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-pro", 
            temperature=0, 
            google_api_key=gemini_api_key
        )
        self.app = self._build_workflow()

    def _retrieve_node(self, state: RAGState) -> RAGState:
        docs = self.retriever.retrieve(query=state["query"], top_k=5)
        return {"context_docs": docs}

    def _generate_node(self, state: RAGState) -> RAGState:
        docs = state["context_docs"]
        
        context_blocks = []
        for d in docs:
            source_info = f"[Source: {d.metadata.get('source_id', 'N/A')}, Element: {d.metadata.get('element_type', 'N/A')}]"
            context_blocks.append(f"{source_info}\n{d.page_content}")
        
        formatted_context = "\n\n".join(context_blocks)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant answering questions grounded strictly in provided context.\n"
                       "Synthesize context from prose, tables, and flowcharts directly. Do not speculate beyond the text."),
            ("human", "Context:\n{context}\n\nQuestion: {query}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"context": formatted_context, "query": state["query"]})
        return {"answer": response.content}

    def _build_workflow(self):
        workflow = StateGraph(RAGState)
        
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    def run(self, query: str) -> str:
        output_state = self.app.invoke({"query": query})
        return output_state["answer"]

# =====================================================================
# STEP 6: INITIALIZATION & GRADIO INTERFACE
# =====================================================================
ingestion_agent = LayoutAwareIngestionAgent()
chunker = GeminiSpecializedChunker(gemini_api_key=GEMINI_KEY)
lancedb_mgr = LanceDBManager(
    table_name="multimodal_rag_poc", 
    db_uri="./lancedb_data", 
    embedding_function=chunker.embeddings
)
pipeline_instance = None

def upload_and_index(file_obj):
    global pipeline_instance
    if file_obj is None:
        return "Please upload a valid file."

    file_path = file_obj.name
    parsed_doc = ingestion_agent.process_file(file_path)
    
    docs = chunker.chunk_prose(
        parsed_doc["markdown_content"], 
        base_metadata={"source_id": parsed_doc["source_id"]}
    )
    
    lancedb_mgr.add_chunks(docs)
    retriever = LanceDBHybridRerankRetriever(lancedb_mgr)
    pipeline_instance = LangGraphLanceDBPipeline(retriever, gemini_api_key=GEMINI_KEY)
    
    return f"Successfully processed '{os.path.basename(file_path)}' into {len(docs)} chunks!"

def ask_question(question: str):
    global pipeline_instance
    if pipeline_instance is None:
        return "Please upload and process a document first before asking questions."
    if not question.strip():
        return "Please enter a valid question."
    
    return pipeline_instance.run(question)

with gr.Blocks(title="Multimodal RAG POC (Gemini + LanceDB)") as demo:
    gr.Markdown("# Multimodal RAG POC Interface")
    gr.Markdown("Upload **PDFs, Word Documents, or PPTX slides** to index with Docling & LanceDB, then ask questions grounded in your document.")

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload Document (.pdf, .docx, .pptx)", file_types=[".pdf", ".docx", ".pptx"])
            upload_button = gr.Button("Process & Index Document", variant="primary")
            status_output = gr.Textbox(label="System Status", interactive=False)
            
        with gr.Column(scale=2):
            question_input = gr.Textbox(label="Your Question", placeholder="e.g., What are the key findings in section 2?", lines=2)
            ask_button = gr.Button("Submit Question", variant="secondary")
            answer_output = gr.Textbox(label="AI Grounded Answer", lines=8, interactive=False)

    upload_button.click(fn=upload_and_index, inputs=[file_input], outputs=[status_output])
    ask_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch(share=True)

