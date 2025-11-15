import pdfplumber
import sys
import json
import re
import os

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "/input/rules.pdf"
output_file = sys.argv[2] if len(sys.argv) > 2 else "/workspace/rules_data.json"

# Get language sections from environment variable or use default
language_sections_json = os.environ.get('LANGUAGE_SECTIONS', '[]')
if language_sections_json == '[]':
    # Default sections for testing
    language_sections_list = [
        ("c", 12, 18, 40, 133),
        ("cpp", 19, 25, 134, 186),  # Extended to page 25 to include STR52-CPP
        ("java", 25, 33, 187, 248),  # Extended to page 33 to include MSC03-J
        ("python", 34, 39, None, None),  # Starts from page 34 (after MSC03-J)
    ]
else:
    language_sections_list = json.loads(language_sections_json)

# Convert to 0-indexed for Python
language_sections = [
    (lang, sum_start - 1, sum_end - 1, 
     det_start - 1 if det_start else None, 
     det_end - 1 if det_end else None)
    for lang, sum_start, sum_end, det_start, det_end in language_sections_list
]

def extract_rules_from_summary(summary_text, language):
    """Extract individual rule entries from summary section."""
    rules = []
    
    # Define expected rule code patterns for each language
    # NOTE: Python uses a different format: PY-SEC-RUL-XX instead of PYT-XX-PY
    language_patterns = {
        'c': r'^[A-Z]+\d+-C$',           # e.g., EXP33-C, ARR30-C
        'cpp': r'^[A-Z]+\d+-CPP$',       # e.g., EXP50-CPP, DCL50-CPP
        'java': r'^[A-Z]+\d+-J$',        # e.g., EXP00-J, MSC03-J
        'python': r'^PY-SEC-RUL-\d+$'    # e.g., PY-SEC-RUL-02, PY-SEC-RUL-04
    }
    
    expected_pattern = language_patterns.get(language, '')
    
    # Pattern to match rule headers - improved to handle different formats
    # Format 1 (C/C++/Java): "3.1.1 EXP33-C. Do not read uninitialized memory"
    # Format 2 (Python): "6.2 PY-SEC-RUL-02. Wildcard injection risk"
    # More flexible pattern that captures the full rule section
    
    # Split by rule headers (section number + rule code)
    # Pattern captures: section_num, rule_code, and everything until next section
    # Updated to match both formats:
    # - Standard: "3.1.1 CODE-X." (e.g., EXP33-C, OBJ01-J) - simple format
    # - Python: "6.2 PY-SEC-RUL-02." (with multiple dashes) - complex format
    # Pattern explanation: [A-Z0-9]+ followed by one or more groups of (-[A-Z0-9]+)
    pattern = r'(\d+(?:\.\d+)*)\s+([A-Z0-9]+(?:-[A-Z0-9]+)+)\.\s+([^\n]+)(.*?)(?=\n\d+(?:\.\d+)*\s+[A-Z0-9]+(?:-[A-Z0-9]+)+\.|\Z)'
    
    for match in re.finditer(pattern, summary_text, re.MULTILINE | re.DOTALL):
        section_num = match.group(1).strip()
        rule_code = match.group(2).strip()
        rule_title = match.group(3).strip()
        content_body = match.group(4).strip()
        
        # FILTER: Only include rules that match the expected language pattern
        if expected_pattern:
            if not re.match(expected_pattern, rule_code):
                print(f"  Skipping {rule_code} (not a {language.upper()} rule, expected pattern {expected_pattern})")
                continue
        
        # Full content includes header and body
        full_content = f"{section_num} {rule_code}. {rule_title}\n{content_body}"
        
        # Extract abstract subsection if present
        abstract_match = re.search(
            r'\d+\.\d+\.\d+\.\d+\s+Abstract\s+(.+?)(?=\d+\.\d+\.\d+\.\d+\s+(?:Risk|Description)|\Z)', 
            full_content, 
            re.DOTALL | re.IGNORECASE
        )
        
        # Extract risk subsection if present
        risk_match = re.search(
            r'\d+\.\d+\.\d+\.\d+\s+Risk\s+(.+?)(?=\d+\.\d+\.\d+\.\d+|\Z)', 
            full_content, 
            re.DOTALL | re.IGNORECASE
        )
        
        # Extract description subsection (for Python which may not have Abstract/Risk)
        desc_match = re.search(
            r'\d+\.\d+\.\d+\.\d+\s+Description\s+(.+?)(?=\d+\.\d+\.\d+\.\d+|\Z)', 
            full_content, 
            re.DOTALL | re.IGNORECASE
        )
        
        abstract_text = abstract_match.group(1).strip() if abstract_match else ''
        risk_text = risk_match.group(1).strip() if risk_match else ''
        desc_text = desc_match.group(1).strip() if desc_match else ''
        
        # If no abstract/risk found, try to extract the content after the title
        if not abstract_text and not risk_text and not desc_text and content_body:
            # Use the content body as description
            desc_text = content_body
        
        rules.append({
            'section': section_num,
            'code': rule_code,
            'title': rule_title,
            'summary': full_content.strip(),
            'abstract': abstract_text,
            'risk': risk_text,
            'description': desc_text,
        })
    
    print(f"  Extracted {len(rules)} rules for {language}")
    if len(rules) > 0:
        print(f"  First rule: {rules[0]['code']} - {rules[0]['title']}")
        print(f"  Last rule: {rules[-1]['code']} - {rules[-1]['title']}")
    
    return rules

def find_detailed_description(details_text, rule_code):
    """Find detailed description for a specific rule code."""
    if not details_text:
        return None
    
    # Pattern to match detailed sections like "A.1 EXP33-C. Do not read uninitialized memory"
    # or "A.1 CTR52-CPP Guarantee that library functions..." (without period)
    # More flexible pattern to handle different appendix numbering and with/without period
    pattern = rf'([A-Z]\.\d+)\s+{re.escape(rule_code)}\.?\s+(.+?)(?=\n[A-Z]\.\d+\s+[A-Z]+\d+-[A-Z]+\.?|\Z)'
    
    match = re.search(pattern, details_text, re.MULTILINE | re.DOTALL)
    if match:
        return match.group(0).strip()
    return None

try:
    with pdfplumber.open(pdf_path) as pdf:
        all_languages_data = {}
        
        for lang, sum_start, sum_end, det_start, det_end in language_sections:
            print(f"\n{'='*60}")
            print(f"Processing {lang.upper()}: pages {sum_start+1}-{sum_end+1}")
            print(f"{'='*60}")
            
            # Extract summary section
            summary_pages = []
            for page_num in range(sum_start, min(sum_end + 1, len(pdf.pages))):
                text = pdf.pages[page_num].extract_text() or ""
                summary_pages.append(text)
            summary_text = "\n\n".join(summary_pages)
            
            # Extract detailed section if exists
            details_text = ""
            if det_start is not None and det_end is not None:
                details_pages = []
                for page_num in range(det_start, min(det_end + 1, len(pdf.pages))):
                    text = pdf.pages[page_num].extract_text() or ""
                    details_pages.append(text)
                details_text = "\n\n".join(details_pages)
            
            # Parse individual rules
            summary_rules = extract_rules_from_summary(summary_text, lang)
            
            # Match with detailed descriptions
            rules_data = []
            for rule in summary_rules:
                detailed = find_detailed_description(details_text, rule['code'])
                rules_data.append({
                    'code': rule['code'],
                    'section': rule['section'],
                    'title': rule['title'],
                    'summary': rule['summary'],
                    'abstract': rule['abstract'],
                    'risk': rule['risk'],
                    'description': rule['description'],
                    'detailed': detailed,
                    'has_detailed': detailed is not None,
                })
            
            all_languages_data[lang] = {
                'rules': rules_data,
                'summary_pages': f"{sum_start + 1}-{sum_end + 1}",
                'details_pages': f"{det_start + 1}-{det_end + 1}" if det_start else "none",
            }
            
            print(f"\nExtracted {lang}: {len(rules_data)} rules from pages {sum_start + 1}-{sum_end + 1}")
            if det_start:
                matched = sum(1 for r in rules_data if r['has_detailed'])
                print(f"  Matched {matched}/{len(rules_data)} with detailed descriptions")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_languages_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Successfully processed {len(all_languages_data)} languages")
    print(f"Output saved to: {output_file}")
    print(f"{'='*60}")
    sys.exit(0)
except Exception as e:
    print(f"Error extracting PDF: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
