#!/usr/bin/env python3
"""Execute system commands in the project directory."""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def execute_command(command: List[str], working_dir: Optional[str] = None) -> tuple[int, str, str]:
    """
    Execute a command in the project directory.
    
    Args:
        command: List of command arguments
        working_dir: Optional working directory (defaults to project root)
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Determine the working directory
    if working_dir is None:
        # Default to the project root (where main.py is located)
        working_dir = Path(__file__).parent.parent.parent.resolve()
    else:
        working_dir = Path(working_dir).resolve()
    
    try:
        # Execute the command
        result = subprocess.run(
            command,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return (result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (1, "", "Command timed out")
    except Exception as e:
        return (1, "", str(e))

def handle_exec_command(args, config):
    """Handle the exec command."""
    from rich.console import Console
    
    console = Console()
    
    # Get the command to execute
    if not args.prompt:
        console.print("[red]No command provided.[/red]")
        return 1
    
    # Parse the command from the prompt
    command_parts = args.prompt.split()
    
    # Validate command
    if not command_parts:
        console.print("[red]Invalid command.[/red]")
        return 1
    
    # Execute command
    return_code, stdout, stderr = execute_command(command_parts)
    
    # Display results
    if return_code == 0:
        if stdout.strip():
            console.print("[green]Command executed successfully:[/green]")
            console.print(stdout)
    else:
        console.print(f"[red]Command failed with exit code {return_code}[/red]")
        if stderr.strip():
            console.print(f"[red]{stderr}[/red]")
    
    return return_code