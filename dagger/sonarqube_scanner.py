#!/usr/bin/env python3
"""
SonarQube Scanner Script for Security Rules Analysis
Runs SonarQube analysis and converts results to SARIF format
"""

import os
import sys
import json
import yaml
import requests
import time
import subprocess
from pathlib import Path

# Disable SSL warnings for internal certificates
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configuration from environment
SONAR_HOST_URL = os.environ.get('SONAR_HOST_URL')
SONAR_TOKEN = os.environ.get('SONAR_TOKEN')
SONAR_PROJECT_KEY = os.environ.get('SONAR_PROJECT_KEY')
SONAR_VERSION = os.environ.get('SONAR_VERSION', '25.9.0.112764')
CREATE_QUALITY_PROFILE = os.environ.get('CREATE_QUALITY_PROFILE', 'false').lower() == 'true'

# Supported languages for Quality Profile creation (PDF rules languages)
SUPPORTED_LANGUAGES = ['c', 'cpp', 'java', 'python']

def load_yaml_rules(rules_path):
    """Load security rules from YAML file."""
    with open(rules_path, 'r') as f:
        rules = yaml.safe_load(f)
    return rules if isinstance(rules, list) else []

def load_rule_mapping(mapping_path='/workspace/rule_mapping.yaml'):
    """Load mapping between PDF rules and SonarQube rules."""
    try:
        with open(mapping_path, 'r') as f:
            mapping = yaml.safe_load(f)
        return mapping if mapping else {}
    except FileNotFoundError:
        print(f"WARNING: Rule mapping file not found at {mapping_path}")
        return {}
    except Exception as e:
        print(f"WARNING: Could not load rule mapping: {e}")
        return {}

def create_quality_profile(language, profile_name, pdf_rules, rule_mapping):
    """
    Create a custom SonarQube Quality Profile with only the rules from PDF.
    
    Args:
        language: Programming language (java, python, cpp, etc.)
        profile_name: Name for the custom quality profile
        pdf_rules: List of PDF rule objects from YAML
        rule_mapping: Mapping dict from rule_mapping.yaml
    
    Returns:
        Profile name if successful, None otherwise
    """
    print(f"\n{'='*60}")
    print(f"Creating Quality Profile: {profile_name}")
    print(f"Language: {language}")
    print(f"{'='*60}")
    
    # Map SonarQube language keys
    lang_map = {'cpp': 'c++', 'csharp': 'cs', 'javascript': 'js'}
    sonar_language = lang_map.get(language.lower(), language.lower())
    
    headers = {'Authorization': f'Bearer {SONAR_TOKEN}'}
    
    # Step 1: Check if profile already exists
    search_url = f"{SONAR_HOST_URL}/api/qualityprofiles/search"
    search_params = {'language': sonar_language}
    
    try:
        response = requests.get(search_url, params=search_params, headers=headers, verify=False)
        response.raise_for_status()
        existing_profiles = response.json().get('profiles', [])
        
        profile_exists = any(p['name'] == profile_name for p in existing_profiles)
        
        if profile_exists:
            print(f"WARNING: Quality Profile '{profile_name}' already exists")
            # Optionally delete and recreate, or just use existing
            # For now, we'll use the existing one
        else:
            # Step 2: Create new quality profile
            create_url = f"{SONAR_HOST_URL}/api/qualityprofiles/create"
            create_params = {
                'language': sonar_language,
                'name': profile_name
            }
            
            response = requests.post(create_url, params=create_params, headers=headers, verify=False)
            if response.status_code == 200:
                print(f"OK: Created Quality Profile: {profile_name}")
            else:
                print(f"WARNING: Warning creating profile: {response.status_code} - {response.text}")
                return None
        
        # Step 3: Deactivate all rules (start with clean slate)
        # Note: This requires getting the profile key first
        profile_key = None
        for p in existing_profiles:
            if p['name'] == profile_name:
                profile_key = p['key']
                break
        
        if not profile_key:
            # Re-fetch profiles after creation
            response = requests.get(search_url, params=search_params, headers=headers, verify=False)
            profiles = response.json().get('profiles', [])
            for p in profiles:
                if p['name'] == profile_name:
                    profile_key = p['key']
                    break
        
        if not profile_key:
            print(f"WARNING: Could not find profile key for '{profile_name}'")
            return None
        
        # Step 4: Activate rules based on mapping
        language_mapping = rule_mapping.get(language, {})
        if not language_mapping:
            print(f"WARNING: No rule mapping found for language '{language}'")
            print(f"   Please add mappings in rule_mapping.yaml")
            return profile_name
        
        activated_count = 0
        for pdf_rule in pdf_rules:
            pdf_rule_id = pdf_rule.get('id', '')
            sonar_rule_keys = language_mapping.get(pdf_rule_id, [])
            
            if not sonar_rule_keys:
                print(f"  ⊘ {pdf_rule_id}: No SonarQube mapping")
                continue
            
            for sonar_key in sonar_rule_keys:
                # Activate rule in profile
                activate_url = f"{SONAR_HOST_URL}/api/qualityprofiles/activate_rule"
                activate_params = {
                    'key': profile_key,
                    'rule': sonar_key
                }
                
                response = requests.post(activate_url, params=activate_params, headers=headers, verify=False)
                if response.status_code == 200:
                    print(f"  OK: {pdf_rule_id} -> {sonar_key}")
                    activated_count += 1
                else:
                    print(f"  WARNING: {pdf_rule_id} -> {sonar_key}: {response.status_code}")
        
        print(f"\nOK: Activated {activated_count} SonarQube rules in profile '{profile_name}'")
        return profile_name
        
    except Exception as e:
        print(f"WARNING: Error creating Quality Profile: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_sonar_scanner(project_path, project_key, quality_profile=None, language=None):
    """Run SonarQube scanner on the project with optional Quality Profile."""
    print(f"\nRunning SonarQube analysis on {project_path}")
    print(f"Project key: {project_key}")
    print(f"SonarQube URL: {SONAR_HOST_URL}")
    
    # Prepare sonar-scanner command
    cmd = [
        'sonar-scanner',
        f'-Dsonar.projectKey={project_key}',
        f'-Dsonar.sources=.',
        f'-Dsonar.host.url={SONAR_HOST_URL}',
        f'-Dsonar.login={SONAR_TOKEN}',
        f'-Dsonar.java.binaries=target/classes',  # For Java projects
        f'-Dsonar.sourceEncoding=UTF-8',
    ]
    
    # Add Quality Profile if specified
    if quality_profile and language:
        cmd.append(f'-Dsonar.qualityprofile.{language}={quality_profile}')
        print(f"Using custom Quality Profile: '{quality_profile}' for {language}")
    
    # Run scanner
    result = subprocess.run(
        cmd,
        cwd=project_path,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"SonarQube scanner failed:")
        print(result.stderr)
        return False
    
    print("OK: SonarQube analysis completed successfully")
    return True

def wait_for_analysis(project_key, max_wait=300):
    """Wait for SonarQube analysis to complete."""
    print("Waiting for SonarQube to process results...")
    
    # Try multiple endpoints in order of preference
    headers = {'Authorization': f'Bearer {SONAR_TOKEN}'}
    
    # First attempt: Check project analyses (usually more permissive)
    analyses_url = f"{SONAR_HOST_URL}/api/project_analyses/search"
    analyses_params = {'project': project_key, 'ps': 1}
    
    start_time = time.time()
    initial_analysis_date = None
    
    while time.time() - start_time < max_wait:
        try:
            # Try project_analyses endpoint (more accessible with project tokens)
            response = requests.get(analyses_url, params=analyses_params, headers=headers, verify=False)
            
            if response.status_code == 200:
                data = response.json()
                analyses = data.get('analyses', [])
                
                if analyses:
                    latest = analyses[0]
                    analysis_date = latest.get('date', '')
                    
                    if not initial_analysis_date:
                        initial_analysis_date = analysis_date
                        print(f"Found analysis from: {analysis_date}")
                        print("Waiting 10 seconds for processing to complete...")
                        time.sleep(10)
                        return True
                    elif analysis_date != initial_analysis_date:
                        # New analysis appeared
                        print(f"OK: New analysis completed: {analysis_date}")
                        return True
                    else:
                        print("Waiting for new analysis...")
                        time.sleep(5)
                else:
                    print("No analyses found yet, waiting...")
                    time.sleep(5)
            
            elif response.status_code == 403:
                # Token doesn't have access, use fixed wait time
                print("WARNING: Cannot check analysis status (403 - insufficient privileges)")
                print("Using fixed wait time of 15 seconds...")
                time.sleep(15)
                return True  # Assume it completed
            
            else:
                print(f"WARNING: Unexpected status {response.status_code}, waiting...")
                time.sleep(5)
                
        except Exception as e:
            print(f"WARNING: Error checking status: {e}")
            print("Using fixed wait time of 15 seconds...")
            time.sleep(15)
            return True  # Assume it completed
    
    print("WARNING: Timeout waiting for analysis, continuing anyway...")
    return True  # Don't fail, just continue

def fetch_sonarqube_issues(project_key):
    """Fetch issues from SonarQube."""
    print(f"Fetching issues for project {project_key}")
    
    url = f"{SONAR_HOST_URL}/api/issues/search"
    headers = {'Authorization': f'Bearer {SONAR_TOKEN}'}
    
    all_issues = []
    page = 1
    page_size = 500
    
    while True:
        params = {
            'componentKeys': project_key,
            'p': page,
            'ps': page_size,
            'resolved': 'false',
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, verify=False)
            response.raise_for_status()
            
            data = response.json()
            issues = data.get('issues', [])
            all_issues.extend(issues)
            
            total = data.get('total', 0)
            print(f"Fetched {len(all_issues)} of {total} issues")
            
            if len(all_issues) >= total:
                break
            
            page += 1
        except Exception as e:
            print(f"Error fetching issues: {e}")
            break
    
    return all_issues

def map_severity(sonar_severity):
    """Map SonarQube severity to SARIF level."""
    mapping = {
        'BLOCKER': 'error',
        'CRITICAL': 'error',
        'MAJOR': 'warning',
        'MINOR': 'note',
        'INFO': 'note',
    }
    return mapping.get(sonar_severity.upper(), 'warning')

def convert_to_sarif(sonar_issues, yaml_rules, project_path):
    """Convert SonarQube issues to SARIF format."""
    print(f"Converting {len(sonar_issues)} issues to SARIF format")
    
    # Build rule index from YAML
    yaml_rules_map = {r['id']: r for r in yaml_rules}
    
    sarif = {
        'version': '2.1.0',
        '$schema': 'https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0-rtm.5.json',
        'runs': [{
            'tool': {
                'driver': {
                    'name': 'SonarQube',
                    'informationUri': SONAR_HOST_URL,
                    'version': SONAR_VERSION,
                    'rules': []
                }
            },
            'results': []
        }]
    }
    
    run = sarif['runs'][0]
    rules_added = set()
    
    for issue in sonar_issues:
        rule_key = issue.get('rule', '')
        component = issue.get('component', '')
        message = issue.get('message', 'Issue detected')
        line = issue.get('line', 1)
        severity = issue.get('severity', 'MAJOR')
        
        # Extract file path from component
        # Component format: "project_key:path/to/file.java"
        file_path = component.split(':', 1)[-1] if ':' in component else component
        
        # Add rule definition if not already added
        if rule_key not in rules_added:
            rule_def = {
                'id': rule_key,
                'name': rule_key,
                'shortDescription': {'text': message[:100]},
                'fullDescription': {'text': message},
                'defaultConfiguration': {'level': map_severity(severity)},
                'properties': {
                    'tags': ['security', 'sonarqube'],
                    'security-severity': str({'BLOCKER': '9.0', 'CRITICAL': '8.0', 'MAJOR': '6.0', 'MINOR': '3.0', 'INFO': '1.0'}.get(severity, '5.0'))
                }
            }
            
            # Check if we have this rule in our YAML
            if rule_key in yaml_rules_map:
                yaml_rule = yaml_rules_map[rule_key]
                rule_def['shortDescription']['text'] = yaml_rule.get('message', message)
                rule_def['fullDescription']['text'] = yaml_rule.get('description', message)
            
            run['tool']['driver']['rules'].append(rule_def)
            rules_added.add(rule_key)
        
        # Add result
        result = {
            'ruleId': rule_key,
            'message': {'text': message},
            'locations': [{
                'physicalLocation': {
                    'artifactLocation': {'uri': file_path},
                    'region': {'startLine': line}
                }
            }]
        }
        
        run['results'].append(result)
    
    print(f"Generated SARIF with {len(rules_added)} rules and {len(run['results'])} results")
    return sarif

def main():
    if len(sys.argv) < 4:
        print("Usage: sonarqube_scanner.py <project_path> <yaml_rules> <output_sarif>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    yaml_rules_path = sys.argv[2]
    output_sarif = sys.argv[3]
    
    # Validate environment
    if not all([SONAR_HOST_URL, SONAR_TOKEN, SONAR_PROJECT_KEY]):
        print("Error: Missing SonarQube configuration in environment")
        print("Required: SONAR_HOST_URL, SONAR_TOKEN, SONAR_PROJECT_KEY")
        sys.exit(1)
    
    print("="*80)
    print("SonarQube Security Rules Compliance Scanner")
    print("="*80)
    
    # Load YAML rules from PDF extraction
    yaml_rules = load_yaml_rules(yaml_rules_path)
    print(f"\nOK: Loaded {len(yaml_rules)} PDF rules from {yaml_rules_path}")
    
    # Detect language from rules
    language = None
    if yaml_rules and len(yaml_rules) > 0:
        language = yaml_rules[0].get('language', '').lower()
        print(f"OK: Detected language: {language}")
    
    if not language:
        print("WARNING: Could not detect language from YAML rules")
        language = 'java'  # Default to Java
    
    # Map to SonarQube language keys
    lang_map = {'cpp': 'c++', 'csharp': 'cs', 'javascript': 'js'}
    sonar_language = lang_map.get(language, language)
    
    # Determine if Quality Profile should be created
    quality_profile = None
    should_create_profile = (
        CREATE_QUALITY_PROFILE and 
        language in SUPPORTED_LANGUAGES
    )
    
    if should_create_profile:
        # Load rule mapping
        rule_mapping = load_rule_mapping()
        
        # Create custom Quality Profile based on PDF rules
        profile_name = f"TAS_Compliance_{language.upper()}_{SONAR_PROJECT_KEY}"
        quality_profile = create_quality_profile(
            language=language,
            profile_name=profile_name,
            pdf_rules=yaml_rules,
            rule_mapping=rule_mapping
        )
        
        if not quality_profile:
            print("\nWARNING: Could not create Quality Profile, using default rules")
    else:
        if not CREATE_QUALITY_PROFILE:
            print(f"\nWARNING: Quality Profile creation disabled (CREATE_QUALITY_PROFILE=false)")
        elif language not in SUPPORTED_LANGUAGES:
            print(f"\nWARNING: Quality Profile not supported for language '{language}'")
            print(f"   Supported languages: {', '.join(SUPPORTED_LANGUAGES)}")
        print("   Using SonarQube default rules")
    
    # Run SonarQube scanner with custom profile
    success = run_sonar_scanner(
        project_path=project_path,
        project_key=SONAR_PROJECT_KEY,
        quality_profile=quality_profile,
        language=sonar_language
    )
    
    if not success:
        print("SonarQube scan failed")
        sys.exit(1)
    
    # Wait for analysis to complete
    if not wait_for_analysis(SONAR_PROJECT_KEY):
        print("WARNING: Analysis may not be complete, continuing anyway...")
    
    # Fetch issues (only those detected by our custom profile)
    issues = fetch_sonarqube_issues(SONAR_PROJECT_KEY)
    
    # Convert to SARIF
    sarif = convert_to_sarif(issues, yaml_rules, project_path)
    
    # Write output
    with open(output_sarif, 'w') as f:
        json.dump(sarif, f, indent=2)
    
    print(f"SARIF report written to {output_sarif}")
    print(f"Total issues: {len(issues)}")

if __name__ == '__main__':
    main()
