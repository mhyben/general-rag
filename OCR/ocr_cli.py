#!/usr/bin/env python3
"""
Interactive OCR CLI application using Marker to convert documents to Markdown.
"""
import os
import sys
from pathlib import Path
from typing import List, Tuple
import gc

# Set environment variables to reduce GPU memory usage
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# Store original CUDA_VISIBLE_DEVICES to restore later if needed
_original_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
except ImportError:
    print("Error: rich library not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import inquirer
except ImportError:
    print("Error: inquirer library not installed. Run: pip install -r requirements.txt")
    sys.exit(1)

# Marker imports will be done lazily after device selection
# This is critical to ensure environment variables are set before models load


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
    
    Args:
        source_file: Source document path
        md_file: Output markdown file path
    
    Note: Environment variables (CUDA_VISIBLE_DEVICES, TORCH_DEVICE) should be
    set before calling this function to control device selection.
    """
    
    try:
        # Import Marker AFTER setting environment variables
        # This ensures models load with the correct device settings
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError:
            console.print("[red]Error: marker-pdf not installed. Run: pip install marker-pdf[/red]")
            return False
        
        # Configure tqdm to update in place properly
        # The issue is likely that tqdm needs proper terminal settings
        try:
            import tqdm
            import sys
            import os
            
            # Ensure tqdm uses stderr (default) and can detect terminal properly
            # Set environment variable to help tqdm detect terminal width
            if 'COLUMNS' not in os.environ:
                try:
                    import shutil
                    columns = shutil.get_terminal_size().columns
                    os.environ['COLUMNS'] = str(columns)
                except:
                    pass
            
            # Configure tqdm to use stderr explicitly and enable dynamic width
            # This helps with in-place updates
            tqdm.tqdm.file = sys.stderr
            # Enable dynamic column width for better terminal compatibility
            if hasattr(tqdm.tqdm, 'dynamic_ncols'):
                tqdm.tqdm.dynamic_ncols = True
        except:
            pass  # If tqdm not available, continue anyway
        
        # Initialize Marker converter
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        
        # Convert document
        rendered = converter(str(source_file))
        text, _, images = text_from_rendered(rendered)
        
        # No need to restore - we didn't patch tqdm, just configured it
        
        # Save markdown
        md_file.write_text(text, encoding='utf-8')
        
        # Clear GPU cache
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        # Force garbage collection
        gc.collect()
        
        return True
    except Exception as e:
        error_msg = str(e)
        console.print(f"[red]Error processing {source_file.name}: {error_msg[:200]}...[/red]")
        
        # If CUDA OOM, suggest CPU mode
        if 'CUDA out of memory' in error_msg:
            console.print(f"[yellow]  Tip: Try running with CPU mode to avoid GPU memory issues[/yellow]")
            console.print(f"[yellow]  Run with: CUDA_VISIBLE_DEVICES=\"\" python ocr_cli.py[/yellow]")
        
        # Clear GPU cache on error
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        
        gc.collect()
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
    
    # Ask about processing mode (CPU vs GPU)
    # IMPORTANT: Set this BEFORE any Marker imports
    use_cpu = False
    if documents_to_process:
        device_options = [
            "Use GPU (faster, requires GPU memory)",
            "Use CPU (slower, no GPU memory required)"
        ]
        
        # Check if CUDA_VISIBLE_DEVICES is set to empty (CPU mode)
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '':
            use_cpu = True
            console.print("[yellow]CPU mode detected (CUDA_VISIBLE_DEVICES=\"\")[/yellow]")
        else:
            questions = [
                inquirer.List(
                    'device',
                    message="Choose processing device:",
                    choices=device_options,
                    default=device_options[1],  # Default to CPU to avoid memory issues
                )
            ]
            
            try:
                answers = inquirer.prompt(questions)
                if answers is None:
                    raise KeyboardInterrupt()
                
                use_cpu = (answers['device'] == device_options[1])
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user[/yellow]")
                sys.exit(0)
        
        # Set environment variables immediately after selection
        # This must happen BEFORE Marker is imported
        if use_cpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['TORCH_DEVICE'] = 'cpu'
            console.print("[dim]CPU mode enabled - GPU will be hidden from Marker[/dim]")
    
    # If there are documents to process, ask user if they want to process all or select specific ones
    if documents_to_process:
        if len(documents_to_process) > 1:
            # Ask user if they want to process all or select specific documents
            process_options = [
                f"Process all {len(documents_to_process)} documents",
                "Select specific documents to process"
            ]
            
            questions = [
                inquirer.List(
                    'action',
                    message="How would you like to proceed?",
                    choices=process_options,
                    default=process_options[0],
                )
            ]
            
            try:
                answers = inquirer.prompt(questions)
                if answers is None:
                    raise KeyboardInterrupt()
                
                if answers['action'] == process_options[1]:
                    # Let user select which documents to process
                    file_choices = [f"{doc[0].name}" for doc in documents_to_process]
                    
                    questions = [
                        inquirer.Checkbox(
                            'files',
                            message="Select documents to process (use space to select, enter to confirm)",
                            choices=file_choices,
                            default=file_choices,  # All selected by default
                        )
                    ]
                    
                    answers = inquirer.prompt(questions)
                    if answers is None:
                        raise KeyboardInterrupt()
                    
                    selected_files = set(answers['files'])
                    documents_to_process = [
                        doc for doc in documents_to_process 
                        if doc[0].name in selected_files
                    ]
            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled by user[/yellow]")
                sys.exit(0)
    
    # Process documents
    if documents_to_process:
        console.print(f"[yellow]Found {len(documents_to_process)} document(s) to process...[/yellow]")
        console.print()
        
        device_mode = "CPU" if use_cpu else "GPU"
        console.print(f"[dim]Processing mode: {device_mode}[/dim]\n")
        
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
                
                # Clear memory between documents
                gc.collect()
                if not use_cpu:
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
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

