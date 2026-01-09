#!/usr/bin/env python3
"""
Interactive OCR CLI application using Marker to convert documents to Markdown.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Error: rich library not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
except ImportError:
    print("Error: marker-pdf not installed. Run: pip install marker-pdf")
    sys.exit(1)


console = Console()


def get_documents_folder() -> Path:
    """Get the documents folder path (project root/documents)."""
    # Get project root (parent of OCR folder)
    project_root = Path(__file__).parent.parent
    documents_folder = project_root / "documents"
    documents_folder.mkdir(exist_ok=True)
    return documents_folder


def find_documents(documents_folder: Path) -> List[Tuple[Path, Path]]:
    """
    Find all documents that need processing.
    Returns list of (source_file, markdown_file) tuples.
    """
    # Supported document extensions
    supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', 
                           '.pptx', '.docx', '.xlsx', '.html', '.epub'}
    
    documents = []
    
    for file_path in documents_folder.iterdir():
        if not file_path.is_file():
            continue
            
        # Check if it's a supported document type
        if file_path.suffix.lower() in supported_extensions:
            # Check if markdown version already exists
            md_file = file_path.with_suffix('.md')
            
            if not md_file.exists():
                documents.append((file_path, md_file))
    
    return documents


def process_document(source_file: Path, md_file: Path) -> bool:
    """
    Convert a document to Markdown using Marker.
    Returns True if successful, False otherwise.
    """
    try:
        # Initialize Marker converter
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        # Convert document
        rendered = converter(str(source_file))
        text, _, images = text_from_rendered(rendered)
        
        # Save markdown
        md_file.write_text(text, encoding='utf-8')
        
        return True
    except Exception as e:
        console.print(f"[red]Error processing {source_file.name}: {str(e)}[/red]")
        return False


def display_results(documents_folder: Path, processed: List[Path], skipped: List[Path], failed: List[Path]):
    """Display processing results in a formatted table."""
    table = Table(title="Processing Results", show_header=True, header_style="bold magenta")
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Status", style="green")
    
    # Add processed files
    for file_path in processed:
        table.add_row(file_path.name, "[green]✓ Processed[/green]")
    
    # Add skipped files (already have .md)
    for file_path in skipped:
        table.add_row(file_path.name, "[yellow]⊘ Skipped (markdown exists)[/yellow]")
    
    # Add failed files
    for file_path in failed:
        table.add_row(file_path.name, "[red]✗ Failed[/red]")
    
    console.print()
    console.print(table)
    console.print()


def main():
    """Main CLI application."""
    console.print(Panel.fit(
        "[bold cyan]OCR Document Converter[/bold cyan]\n"
        "Converts documents to Markdown using Marker",
        border_style="cyan"
    ))
    console.print()
    
    # Get documents folder
    documents_folder = get_documents_folder()
    console.print(f"[dim]Documents folder: {documents_folder}[/dim]")
    console.print()
    
    # Find all documents
    documents_to_process = find_documents(documents_folder)
    
    # Find all files in documents folder (for display)
    all_files = [f for f in documents_folder.iterdir() if f.is_file()]
    all_doc_files = [f for f in all_files if f.suffix.lower() in 
                     {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.tif', 
                      '.pptx', '.docx', '.xlsx', '.html', '.epub', '.md'}]
    
    # Separate files into categories
    processed = []
    skipped = []
    failed = []
    
    # Files that already have markdown versions
    for file_path in all_doc_files:
        if file_path.suffix.lower() == '.md':
            continue
        md_file = file_path.with_suffix('.md')
        if md_file.exists():
            skipped.append(file_path)
    
    # Process documents
    if documents_to_process:
        console.print(f"[yellow]Found {len(documents_to_process)} document(s) to process...[/yellow]")
        console.print()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            for source_file, md_file in documents_to_process:
                task = progress.add_task(f"Processing {source_file.name}...", total=None)
                
                success = process_document(source_file, md_file)
                
                if success:
                    processed.append(source_file)
                else:
                    failed.append(source_file)
                
                progress.remove_task(task)
    else:
        console.print("[green]No documents to process. All documents already have markdown versions.[/green]")
        console.print()
    
    # Display results
    if processed or skipped or failed:
        display_results(documents_folder, processed, skipped, failed)
    
    # Summary
    console.print(Panel(
        f"[bold]Summary:[/bold]\n"
        f"  • Processed: [green]{len(processed)}[/green]\n"
        f"  • Skipped: [yellow]{len(skipped)}[/yellow]\n"
        f"  • Failed: [red]{len(failed)}[/red]",
        border_style="blue"
    ))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")
        sys.exit(1)

