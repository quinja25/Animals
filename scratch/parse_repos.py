import re

def parse_md(filepath):
    print(f"\n=== Parsing {filepath} ===")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Let's search for python files, .py or .ipynb
    py_files = set(re.findall(r'href="[^"]+\.py"', content))
    ipynb_files = set(re.findall(r'href="[^"]+\.ipynb"', content))
    md_files = set(re.findall(r'href="[^"]+\.md"', content))
    
    print("Python files found in links:")
    for py in py_files:
        print(" ", py)
    print("Notebooks found:")
    for ip in ipynb_files:
        print(" ", ip)
    print("Markdown files found:")
    for md in md_files:
        print(" ", md)

parse_md("C:/Users/jaeyo/.gemini/antigravity-cli/brain/05186fea-be0e-45e3-935d-75dccfde68a3/.system_generated/steps/14/content.md")
parse_md("C:/Users/jaeyo/.gemini/antigravity-cli/brain/05186fea-be0e-45e3-935d-75dccfde68a3/.system_generated/steps/15/content.md")
