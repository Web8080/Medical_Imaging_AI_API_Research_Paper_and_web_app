#!/usr/bin/env python3
"""
Convert LaTeX research paper to Markdown format
"""

import re
import sys

def convert_latex_to_markdown(tex_content):
    """Convert LaTeX to Markdown"""
    md = tex_content
    
    # Remove LaTeX preamble
    md = re.sub(r'\\documentclass.*?\\begin\{document\}', '', md, flags=re.DOTALL)
    md = re.sub(r'\\end\{document\}', '', md)
    
    # Remove maketitle, tableofcontents, newpage
    md = re.sub(r'\\maketitle', '', md)
    md = re.sub(r'\\tableofcontents', '', md)
    md = re.sub(r'\\newpage', '', md)
    
    # Convert sections
    md = re.sub(r'\\section\{([^}]+)\}', r'\n## \1\n', md)
    md = re.sub(r'\\subsection\{([^}]+)\}', r'\n### \1\n', md)
    md = re.sub(r'\\subsubsection\{([^}]+)\}', r'\n#### \1\n', md)
    
    # Convert text formatting
    md = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', md)
    md = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', md)
    md = re.sub(r'\\texttt\{([^}]+)\}', r'`\1`', md)
    
    # Convert lists
    md = re.sub(r'\\begin\{enumerate\}', '', md)
    md = re.sub(r'\\end\{enumerate\}', '', md)
    md = re.sub(r'\\begin\{itemize\}', '', md)
    md = re.sub(r'\\end\{itemize\}', '', md)
    md = re.sub(r'^\s*\\item\s+', '- ', md, flags=re.MULTILINE)
    
    # Convert abstract
    md = re.sub(r'\\begin\{abstract\}', '\n## Abstract\n', md)
    md = re.sub(r'\\end\{abstract\}', '', md)
    
    # Convert tables
    md = re.sub(r'\\begin\{table\}.*?\\end\{table\}', '[Table - See LaTeX version for formatting]', md, flags=re.DOTALL)
    
    # Convert citations
    md = re.sub(r'\\cite\{([^}]+)\}', r'[\1]', md)
    
    # Convert URLs
    md = re.sub(r'\\url\{([^}]+)\}', r'[\1](\1)', md)
    
    # Convert bibliography
    md = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}', '\n## References\n\n[See LaTeX version for complete bibliography]\n', md, flags=re.DOTALL)
    
    # Remove remaining LaTeX commands
    md = re.sub(r'\\[a-zA-Z]+\*?\{([^}]*)\}', r'\1', md)
    md = re.sub(r'\\[a-zA-Z]+\*?', '', md)
    
    # Clean up multiple blank lines
    md = re.sub(r'\n\n\n+', '\n\n', md)
    
    return md.strip()

if __name__ == '__main__':
    # Read LaTeX file
    with open('docs/research_paper/Medical_Imaging_AI_API_Research_Paper.tex', 'r') as f:
        tex_content = f.read()
    
    # Convert to Markdown
    md_content = convert_latex_to_markdown(tex_content)
    
    # Write Markdown file
    with open('docs/research_paper/Medical_Imaging_AI_API_Research_Paper.md', 'w') as f:
        f.write(md_content)
    
    print("Conversion complete!")
    print(f"Markdown file created: docs/research_paper/Medical_Imaging_AI_API_Research_Paper.md")

