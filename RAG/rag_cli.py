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

try:
    import inquirer
except ImportError:
    print("Error: inquirer library not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Import RAG components
from embedding_loader import EmbeddingLoader
from agent_llm import AgentLLM
from vector_store import FAISSVectorStore
from structure_aware_chunking import StructureAwareChunker

# Import all strategies using importlib for modules with numbers
import importlib
from strategies.base_strategy import BaseRAGStrategy

# Map strategy keys to module names and class names
STRATEGY_MODULES = {
    "01_reranking": ("strategies.01_reranking", "RerankingStrategy"),
    "02_agentic_rag": ("strategies.02_agentic_rag", "AgenticRAGStrategy"),
    "03_knowledge_graphs": ("strategies.03_knowledge_graphs", "KnowledgeGraphsStrategy"),
    "04_contextual_retrieval": ("strategies.04_contextual_retrieval", "ContextualRetrievalStrategy"),
    "05_query_expansion": ("strategies.05_query_expansion", "QueryExpansionStrategy"),
    "06_multi_query_rag": ("strategies.06_multi_query_rag", "MultiQueryRAGStrategy"),
    "07_context_aware_chunking": ("strategies.07_context_aware_chunking", "ContextAwareChunkingStrategy"),
    "08_late_chunking": ("strategies.08_late_chunking", "LateChunkingStrategy"),
    "09_hierarchical_rag": ("strategies.09_hierarchical_rag", "HierarchicalRAGStrategy"),
    "10_self_reflective_rag": ("strategies.10_self_reflective_rag", "SelfReflectiveRAGStrategy"),
    "11_fine_tuned_embeddings": ("strategies.11_fine_tuned_embeddings", "FineTunedEmbeddingsStrategy"),
}


console = Console()


def select_option(prompt: str, options: List[str], default: Optional[str] = None) -> int:
    """
    Interactive selection with cursor navigation using inquirer.
    
    Args:
        prompt: Prompt text
        options: List of option strings
        default: Default option (if None, uses first option)
        
    Returns:
        Index of selected option
    """
    if default is None and options:
        default = options[0]
    
    questions = [
        inquirer.List(
            'choice',
            message=prompt,
            choices=options,
            default=default,
        )
    ]
    
    try:
        answers = inquirer.prompt(questions)
        if answers is None:
            # User cancelled (Ctrl+C)
            raise KeyboardInterrupt()
        
        selected = answers['choice']
        return options.index(selected)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        console.print(f"[red]Selection error: {str(e)}[/red]")
        raise


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


def load_documents(vector_store: FAISSVectorStore, embedder: EmbeddingLoader, 
                  folder: Path, chunker: StructureAwareChunker):
    """Load markdown documents into the vector store using structure-aware chunking."""
    md_files = find_markdown_files(folder)
    
    if not md_files:
        console.print(f"[yellow]No markdown files found in {folder}[/yellow]")
        return
    
    console.print(f"[cyan]Loading {len(md_files)} document(s) with structure-aware chunking...[/cyan]")
    
    all_chunks = []
    all_metadatas = []
    
    for md_file in md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            
            # Use structure-aware chunking
            chunk_data = chunker.chunk_document(content, source=md_file.name)
            
            # Create metadata for each chunk
            for i, chunk_info in enumerate(chunk_data):
                chunk_text = chunk_info['content']
                
                # Add context to chunk text if available
                if chunk_info.get('context'):
                    contextualized_chunk = f"[Context: {chunk_info['context']}]\n\n{chunk_text}"
                elif chunk_info.get('heading'):
                    contextualized_chunk = f"[Section: {chunk_info['heading']}]\n\n{chunk_text}"
                else:
                    contextualized_chunk = chunk_text
                
                all_chunks.append(contextualized_chunk)
                all_metadatas.append({
                    'content': chunk_text,  # Store original content
                    'contextualized_content': contextualized_chunk,  # Store with context
                    'source': md_file.name,
                    'chunk_id': i,
                    'file_path': str(md_file),
                    'context': chunk_info.get('context', ''),
                    'heading': chunk_info.get('heading', ''),
                    'heading_level': chunk_info.get('heading_level', 0)
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
    if name not in STRATEGY_MODULES:
        raise ValueError(f"Unknown strategy: {name}")
    
    module_name, class_name = STRATEGY_MODULES[name]
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise RuntimeError(f"Failed to load strategy '{name}': {str(e)}")


def main():
    """Main CLI application."""
    console.print(Panel.fit(
        "[bold cyan]RAG Application[/bold cyan]\n"
        "Retrieval-Augmented Generation with Multiple Strategies",
        border_style="cyan"
    ))
    
    # Step 1: Select embedding model
    console.print("\n[bold]Step 1: Select Embedding Model[/bold]")
    popular_models = EmbeddingLoader.list_popular_models()
    model_idx = select_option("Choose an embedding model:", popular_models)
    selected_model = popular_models[model_idx]
    
    console.print(f"[green]Selected: {selected_model}[/green]")
    console.print("[dim]Loading embedding model...[/dim]")
    
    try:
        embedder = EmbeddingLoader(selected_model)
        console.print("[green]✓ Embedding model loaded[/green]")
    except Exception as e:
        error_msg = str(e)
        console.print(f"[red]Failed to load embedding model: {error_msg}[/red]")
        
        # Check for PyTorch CUDA timing error
        if "fast_1 >= fast_0" in error_msg or "INTERNAL ASSERT FAILED" in error_msg:
            console.print("\n[yellow]PyTorch CUDA timing error detected![/yellow]")
            console.print("[yellow]Try one of these solutions:[/yellow]")
            console.print("  1. Run with CPU-only: [cyan]export CUDA_VISIBLE_DEVICES=\"\" && python rag_cli.py[/cyan]")
            console.print("  2. See TROUBLESHOOTING.md for more solutions")
        
        sys.exit(1)
    
    # Step 2: Select Ollama LLM
    console.print("\n[bold]Step 2: Select Ollama LLM Model[/bold]")
    
    try:
        agent_llm = AgentLLM()
        available_models = agent_llm.list_models()
        
        if not available_models:
            console.print("[red]No Ollama models found. Please install models using: ollama pull <model_name>[/red]")
            sys.exit(1)
        
        llm_idx = select_option("Choose an Ollama model:", available_models)
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
    
    strategy_idx = select_option("Choose a RAG strategy:", strategy_names)
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
    
    # Step 5: Configure chunking parameters (optional for structure-aware chunking)
    console.print("\n[bold]Step 5: Configure Chunking Parameters (Optional)[/bold]")
    console.print("[dim]Structure-aware chunking uses document structure (headings, sections) to determine chunk boundaries.[/dim]")
    console.print("[dim]These parameters are optional limits for very large sections.[/dim]\n")
    
    configure_options = ["Use defaults (structure-only, no size limits)", "Configure maximum chunk size"]
    config_idx = select_option("Chunking configuration:", configure_options)
    
    max_chunk_size = None
    overlap = None
    
    if config_idx == 1:  # User wants to configure
        chunk_size_options = ["500", "800", "1000", "1500", "2000", "No limit", "Custom"]
        chunk_size_idx = select_option("Maximum chunk size (characters, for very large sections):", chunk_size_options)
        
        if chunk_size_options[chunk_size_idx] == "No limit":
            max_chunk_size = None
        elif chunk_size_options[chunk_size_idx] == "Custom":
            chunk_size_str = Prompt.ask("Enter custom maximum chunk size", default="1000")
            try:
                max_chunk_size = int(chunk_size_str) if chunk_size_str else None
            except ValueError:
                console.print("[yellow]Invalid input, using no limit[/yellow]")
                max_chunk_size = None
        else:
            max_chunk_size = int(chunk_size_options[chunk_size_idx])
        
        # Overlap is rarely needed with structure-aware chunking
        overlap_options = ["No overlap (recommended)", "50", "100", "150", "Custom"]
        overlap_idx = select_option("Chunk overlap (usually not needed with structure-aware):", overlap_options)
        
        if overlap_options[overlap_idx] == "No overlap (recommended)":
            overlap = None
        elif overlap_options[overlap_idx] == "Custom":
            overlap_str = Prompt.ask("Enter custom overlap", default="0")
            try:
                overlap = int(overlap_str) if overlap_str else None
            except ValueError:
                overlap = None
        else:
            overlap = int(overlap_options[overlap_idx])
    
    if max_chunk_size:
        console.print(f"[green]Maximum chunk size: {max_chunk_size}[/green]")
    else:
        console.print("[green]No chunk size limit (structure-only chunking)[/green]")
    
    if overlap:
        console.print(f"[green]Overlap: {overlap}[/green]")
    else:
        console.print("[green]No overlap (natural structure boundaries)[/green]")
    
    # Initialize structure-aware chunker
    chunker = StructureAwareChunker(max_chunk_size=max_chunk_size, overlap=overlap)
    
    # Initialize vector store
    dimension = embedder.get_embedding_dimension()
    vector_store = FAISSVectorStore(dimension)
    
    # Load documents with structure-aware chunking
    load_documents(vector_store, embedder, documents_folder, chunker)
    
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

