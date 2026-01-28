"""Language detection and extension mapping."""

import os
from pathlib import Path
from typing import Optional

# Extension to language mapping
EXT_TO_LANGUAGE = {
    # Python
    '.py': 'python',
    '.pyw': 'python',
    # JavaScript/TypeScript
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.ts': 'typescript',
    '.tsx': 'typescript',
    # Java
    '.java': 'java',
    # Go
    '.go': 'go',
    # Rust
    '.rs': 'rust',
    # C/C++
    '.c': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.h': 'c',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    # C#
    '.cs': 'csharp',
    # Kotlin
    '.kt': 'kotlin',
    # Swift
    '.swift': 'swift',
    # PHP
    '.php': 'php',
    # Ruby
    '.rb': 'ruby',
    # Scala
    '.scala': 'scala',
    # Markdown
    '.md': 'markdown',
    '.markdown': 'markdown',
    # Shell
    '.sh': 'shell',
    '.bash': 'shell',
    # YAML
    '.yml': 'yaml',
    '.yaml': 'yaml',
    # JSON
    '.json': 'json',
    # HTML/CSS
    '.html': 'html',
    '.htm': 'html',
    '.css': 'css',
    '.scss': 'scss',
    '.sass': 'sass',
    # XML
    '.xml': 'xml',
    # SQL
    '.sql': 'sql',
    # R
    '.r': 'r',
    '.R': 'r',
}

# Supported languages with AST parsing
AST_SUPPORTED_LANGUAGES = {'python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'c'}

# Languages that should use simple chunking fallback
SIMPLE_CHUNK_LANGUAGES = {'markdown', 'shell', 'yaml', 'json', 'html', 'css'}


def detect_language(filepath: str) -> Optional[str]:
    """
    Detect language from file extension or shebang.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Language identifier or None if unknown
    """
    # Try extension first
    ext = Path(filepath).suffix.lower()
    if ext in EXT_TO_LANGUAGE:
        return EXT_TO_LANGUAGE[ext]

    # Try shebang for executable files
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            if first_line.startswith('#!'):
                shebang = first_line.lower()
                if 'python' in shebang:
                    return 'python'
                elif 'node' in shebang or 'javascript' in shebang:
                    return 'javascript'
                elif 'ruby' in shebang:
                    return 'ruby'
                elif 'bash' in shebang or 'sh' in shebang:
                    return 'shell'
    except:
        pass

    return None


def supports_ast_parsing(language: str) -> bool:
    """Check if language has AST parser support."""
    return language in AST_SUPPORTED_LANGUAGES


def is_simple_chunk_language(language: str) -> bool:
    """Check if language should use simple chunking."""
    return language in SIMPLE_CHUNK_LANGUAGES


def get_file_extension(filepath: str) -> str:
    """Get file extension with dot."""
    return Path(filepath).suffix.lower()


def get_chunk_type_display(chunk_type: str) -> str:
    """Get human-readable chunk type."""
    type_names = {
        'function': 'Function',
        'class': 'Class',
        'method': 'Method',
        'interface': 'Interface',
        'module': 'Module',
        'snippet': 'Code Snippet',
    }
    return type_names.get(chunk_type, 'Snippet')
