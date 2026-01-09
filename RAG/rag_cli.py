#!/usr/bin/env python3
"""
Interactive RAG CLI application with cursor navigation.
Allows users to select embedding models, LLM models, RAG strategies, and ask questions.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import re

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich.table import Table
    from rich.markdown import Markdown
    from rich import box
except ImportError:
    print("Error: rich library not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Import RAG components
from embedding_loader import EmbeddingLoader
from agent_llm import AgentLLM
from vector_store import FAISSVectorStore

# Import all strategies
from strategies.base_strategy import BaseRAGStrategy
from strategies import (
    reranking as reranking_module,
    agentic_rag as agentic_rag_module,
    knowledge_graphs as knowledge_graphs_module,
    contextual_retrieval as contextual_retrieval_module,
    query_expansion as query_expansion_module,
    multi_query_rag as multi_query_rag_module,
    context_aware_chunking as context_aware_chunking_module,
    late_chunking as late_chunking_module,
    hierarchical_rag as hierarchical_rag_module,
    self_reflective_rag as self_reflective_rag_module,
    fine_tuned_embeddings as fine_tuned_embeddings_module,
)

# Map strategy keys to classes
STRATEGY_CLASSES = {
    "01_reranking": reranking_module.RerankingStrategy,
    "02_agentic_rag": agentic_rag_module.AgenticRAGStrategy,
    "03_knowledge_graphs": knowledge_graphs_module.KnowledgeGraphsStrategy,
    "04_contextual_retrieval": contextual_retrieval_module.ContextualRetrievalStrategy,
    "05_query_expansion": query_expansion_module.QueryExpansionStrategy,
    "06_multi_query_rag": multi_query_rag_module.MultiQueryRAGStrategy,
    "07_context_aware_chunking": context_aware_chunking_module.ContextAwareChunkingStrategy,
    "08_late_chunking": late_chunking_module.LateChunkingStrategy,
    "09_hierarchical_rag": hierarchical_rag_module.HierarchicalRAGStrategy,
    "10_self_reflective_rag": self_reflective_rag_module.SelfReflectiveRAGStrategy,
    "11_fine_tuned_embeddings": fine_tuned_embeddings_module.FineTunedEmbeddingsStrategy,
}


console = Console()


class InteractiveSelector:
    """Interactive selector with cursor navigation."""
    
    def __init__(self, console: Console):
        self.console = console
    
    def select(self, prompt: str, options: List[str], default: int = 0) -> int:
        """
        Interactive selection with arrow keys.
        Simplified version using rich Prompt for now.
        """
        self.console.print(f"\n[cyan]{prompt}[/cyan]")
        
        # Display options
        table = Table(box=box.SIMPLE, show_header=False)
        for i, option in enumerate(options):
            marker = "→" if i == default else " "
            table.add_row(f"{marker} {i+1}. {option}")
        
        self.console.print(table)
        
        # Get selection
        while True:
            try:
                choice = Prompt.ask(
                    f"\nSelect option (1-{len(options)})",
                    default=str(default + 1)
                )
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return idx
                else:
                    self.console.print("[red]Invalid selection. Please try again.[/red]")
            except (ValueError, KeyboardInterrupt):
                self.console.print("[red]Invalid input. Please enter a number.[/red]")


def get_documents_folder() -> Path:
    """Get the documents folder path."""
    project_root = Path(__file__).parent.parent
    documents_folder = project_root / "documents"
    documents_folder.mkdir(exist_ok=True)
    return documents_folder


def find_markdown_files(folder: Path) -> List[Path]:
    """Find all markdown files in a folder."""
    if not folder.exists():
        return []
    return list(folder.glob("*.md"))


def load_documents(vector_store: FAISSVectorStore, embedder: EmbeddingLoader, folder: Path):
    """Load markdown documents into the vector store."""
    md_files = find_markdown_files(folder)
    
    if not md_files:
        console.print(f"[yellow]No markdown files found in {folder}[/yellow]")
        return
    
    console.print(f"[cyan]Loading {len(md_files)} document(s)...[/cyan]")
    
    all_chunks = []
    all_metadatas = []
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            
            # Simple chunking (500 chars with overlap)
            chunk_size = 500
            overlap = 100
            
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                chunks.append(chunk)
                start = end - overlap
            
            # Create metadata for each chunk
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    'content': chunk,
                    'source': md_file.name,
                    'chunk_id': i,
                    'file_path': str(md_file)
                })
        except Exception as e:
            console.print(f"[red]Error loading {md_file.name}: {str(e)}[/red]")
    
    if all_chunks:
        # Embed all chunks
        console.print("[cyan]Embedding documents...[/cyan]")
        embeddings = embedder.embed_documents(all_chunks)
        
        # Add to vector store
        vector_store.add_documents(embeddings, all_metadatas)
        console.print(f"[green]✓ Loaded {len(all_chunks)} chunks from {len(md_files)} document(s)[/green]\n")


def get_strategy_class(name: str) -> type:
    """Get strategy class by name."""
    return STRATEGY_CLASSES.get(name)


def main():
    """Main CLI application."""
    console.print(Panel.fit(
        "[bold cyan]RAG Application[/bold cyan]\n"
        "Retrieval-Augmented Generation with Multiple Strategies",
        border_style="cyan"
    ))
    
    selector = InteractiveSelector(console)
    
    # Step 1: Select embedding model
    console.print("\n[bold]Step 1: Select Embedding Model[/bold]")
    popular_models = EmbeddingLoader.list_popular_models()
    model_idx = selector.select("Choose an embedding model:", popular_models)
    selected_model = popular_models[model_idx]
    
    console.print(f"[green]Selected: {selected_model}[/green]")
    console.print("[dim]Loading embedding model...[/dim]")
    
    try:
        embedder = EmbeddingLoader(selected_model)
        console.print("[green]✓ Embedding model loaded[/green]")
    except Exception as e:
        console.print(f"[red]Failed to load embedding model: {str(e)}[/red]")
        sys.exit(1)
    
    # Step 2: Select Ollama LLM
    console.print("\n[bold]Step 2: Select Ollama LLM Model[/bold]")
    
    try:
        agent_llm = AgentLLM()
        available_models = agent_llm.list_models()
        
        if not available_models:
            console.print("[red]No Ollama models found. Please install models using: ollama pull <model_name>[/red]")
            sys.exit(1)
        
        llm_idx = selector.select("Choose an Ollama model:", available_models)
        selected_llm = available_models[llm_idx]
        
        agent_llm.set_model(selected_llm)
        console.print(f"[green]Selected: {selected_llm}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to connect to Ollama: {str(e)}[/red]")
        sys.exit(1)
    
    # Step 3: Select RAG strategy
    console.print("\n[bold]Step 3: Select RAG Strategy[/bold]")
    
    strategy_names = [
        "01. Re-ranking",
        "02. Agentic RAG",
        "03. Knowledge Graphs",
        "04. Contextual Retrieval",
        "05. Query Expansion",
        "06. Multi-Query RAG",
        "07. Context-Aware Chunking",
        "08. Late Chunking",
        "09. Hierarchical RAG",
        "10. Self-Reflective RAG",
        "11. Fine-tuned Embeddings",
    ]
    
    strategy_keys = [
        "01_reranking",
        "02_agentic_rag",
        "03_knowledge_graphs",
        "04_contextual_retrieval",
        "05_query_expansion",
        "06_multi_query_rag",
        "07_context_aware_chunking",
        "08_late_chunking",
        "09_hierarchical_rag",
        "10_self_reflective_rag",
        "11_fine_tuned_embeddings",
    ]
    
    strategy_idx = selector.select("Choose a RAG strategy:", strategy_names)
    selected_strategy_key = strategy_keys[strategy_idx]
    selected_strategy_name = strategy_names[strategy_idx]
    
    console.print(f"[green]Selected: {selected_strategy_name}[/green]")
    
    # Step 4: Select documents folder
    console.print("\n[bold]Step 4: Select Documents Folder[/bold]")
    
    default_folder = get_documents_folder()
    folder_path_str = Prompt.ask(
        "Enter path to documents folder (markdown files)",
        default=str(default_folder)
    )
    
    documents_folder = Path(folder_path_str)
    
    if not documents_folder.exists():
        console.print(f"[red]Folder not found: {documents_folder}[/red]")
        sys.exit(1)
    
    console.print(f"[green]Using folder: {documents_folder}[/green]")
    
    # Initialize vector store
    dimension = embedder.get_embedding_dimension()
    vector_store = FAISSVectorStore(dimension)
    
    # Load documents
    load_documents(vector_store, embedder, documents_folder)
    
    if vector_store.get_size() == 0:
        console.print("[red]No documents loaded. Exiting.[/red]")
        sys.exit(1)
    
    # Initialize strategy
    try:
        strategy_class = get_strategy_class(selected_strategy_key)
        
        # Special handling for agentic RAG (needs documents folder)
        if selected_strategy_key == "02_agentic_rag":
            strategy = strategy_class(embedder, agent_llm, vector_store, documents_folder=str(documents_folder))
        else:
            strategy = strategy_class(embedder, agent_llm, vector_store)
        
        console.print(f"[green]✓ Strategy initialized[/green]")
    except Exception as e:
        console.print(f"[red]Failed to initialize strategy: {str(e)}[/red]")
        sys.exit(1)
    
    # Q&A Loop
    console.print("\n[bold cyan]Ready! Ask questions (type '/bye' or 'exit' to quit)[/bold cyan]\n")
    
    while True:
        try:
            query = Prompt.ask("[bold]Your question[/bold]")
            
            if query.lower() in ['/bye', 'exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break
            
            if not query.strip():
                continue
            
            console.print("\n[dim]Thinking...[/dim]")
            
            try:
                answer = strategy.answer(query)
                
                console.print("\n[bold green]Answer:[/bold green]")
                console.print(Markdown(answer))
                console.print()
            except Exception as e:
                console.print(f"[red]Error generating answer: {str(e)}[/red]\n")
        
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Type '/bye' or 'exit' to quit.[/yellow]\n")
        except EOFError:
            console.print("\n[yellow]Goodbye![/yellow]")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)

