import json
import os
import html
import argparse
from typing import Dict, Any, List

# --- Sev utility and JSON loader ---
def sev_val(s: str) -> int:
    m = {'CRITICAL': 5,'HIGH': 4,'MEDIUM': 3,'LOW': 2,'INFO': 1,'UNKNOWN': 0}
    return m.get((s or '').upper(), 0)

def load_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# --- Vulnerabilities extraction ---
def extract_vulnerabilities(sbom: dict, thr_v: int):
    vulns: List[Dict[str, Any]] = []
    if not isinstance(sbom, dict):
        return vulns

    components = {c.get('bom-ref'): c for c in sbom.get('components', []) or [] if c.get('bom-ref')}
    
    # Top-level vulnerabilities
    for v in sbom.get('vulnerabilities', []) or []:
        sev = 'UNKNOWN'
        ratings = v.get('ratings') or []
        for r in ratings:
            if 'CVSSV3' in (r.get('method') or '').upper():
                sev = (r.get('severity') or '').upper() or sev
                break
        if sev == 'UNKNOWN':
            for r in ratings:
                src = r.get('source') or {}
                if isinstance(src, dict) and src.get('name','').lower() == 'nvd':
                    sev = (r.get('severity') or '').upper() or sev
                    break
        if sev == 'UNKNOWN' and ratings:
            best = max((sev_val(r.get('severity') or '') for r in ratings), default=0)
            for name, val in [('CRITICAL',5),('HIGH',4),('MEDIUM',3),('LOW',2),('INFO',1)]:
                if val == best:
                    sev = name
                    break
        if sev == 'UNKNOWN':
            sev = (v.get('severity') or v.get('severityName') or 'UNKNOWN').upper()
        
        if sev_val(sev) >= thr_v:
            vid = v.get('id') or ''
            desc = v.get('description') or v.get('detail') or ''
            pkg = ''
            affects = v.get('affects') or []
            if isinstance(affects,list) and affects:
                ref = affects[0].get('ref') if isinstance(affects[0],dict) else affects[0]
                if isinstance(ref,str):
                    comp = components.get(ref)
                    pkg = comp.get('name') or comp.get('purl') or ref if comp else ref
            if not pkg:
                raw_pkg = v.get('component') or v.get('module') or v.get('package') or ''
                if isinstance(raw_pkg, dict):
                    pkg = raw_pkg.get('name') or raw_pkg.get('bom-ref') or str(raw_pkg)
                else:
                    pkg = raw_pkg or v.get('module') or v.get('package') or vid
            vulns.append({'id': vid,'package': pkg,'severity': sev,'description': desc})
    return vulns

# --- SARIF errors ---
def extract_sarif_errors(sarif: dict):
    errors: List[Dict[str, Any]] = []
    if not isinstance(sarif, dict):
        return errors
    for run in sarif.get('runs', []):
        rule_level = {}
        driver = run.get('tool',{}).get('driver',{})
        for rule in driver.get('rules',[]) or []:
            rid = rule.get('id')
            lvl = (rule.get('defaultConfiguration',{}).get('level') or '').lower()
            if rid:
                rule_level[rid] = lvl
        for res in run.get('results',[]) or []:
            level = (res.get('level') or '').lower() or rule_level.get(res.get('ruleId'),'')
            if level == 'error':
                msg = res.get('message',{})
                text = msg.get('text') if isinstance(msg,dict) else str(msg)
                file_line = ''
                locs = res.get('locations',[]) or []
                if locs:
                    try:
                        pl = locs[0].get('physicalLocation',{})
                        art = pl.get('artifactLocation',{})
                        uri = art.get('uri') or ''
                        region = pl.get('region',{})
                        start = region.get('startLine') or region.get('startColumn')
                        if uri:
                            file_line = f"{uri}:{start}" if start else uri
                    except Exception:
                        pass
                errors.append({'rule': res.get('ruleId'),'message': text,'locations': locs,'file_line': file_line})
    return errors

# --- HTML writer ---
def write_html(path: str, summary: Dict[str,Any]):
    vulns = summary.get('vulnerabilities',[])
    sonar_errors = summary.get('sonar_errors',[])
    gg_errors = summary.get('gg_errors',[])
    counts = summary.get('counts',{})

    with open(path,'w',encoding='utf-8') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Synthetic Report</title>')
        f.write('<style>table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f2f2f2;text-align:left}</style>')
        f.write('</head><body>')
        f.write('<h1>Synthetic Report</h1>')

        if summary.get('base_image') or summary.get('operating_system'):
            f.write('<h2>Environment</h2><table><tbody>')
            if summary.get('base_image'):
                f.write(f'<tr><th>Base image</th><td>{html.escape(str(summary["base_image"]))}</td></tr>')
            if summary.get('operating_system'):
                f.write(f'<tr><th>Operating system</th><td>{html.escape(str(summary["operating_system"]))}</td></tr>')
            f.write('</tbody></table>')

        f.write('<h2>Counts</h2><table><thead><tr><th>Metric</th><th>Count</th></tr></thead><tbody>')
        for k in ['CRITICAL','HIGH','MEDIUM','LOW','INFO','UNKNOWN']:
            c = counts.get(k,0)
            if c>0:
                f.write(f'<tr><td>VULNERABILITIES - {k}</td><td>{c}</td></tr>')
        f.write(f'<tr><td>SOURCE CODE - SonarQube Errors</td><td>{len(sonar_errors)}</td></tr>')
        f.write(f'<tr><td>SOURCE CODE - GitGuardian Errors</td><td>{len(gg_errors)}</td></tr>')
        f.write('</tbody></table>')

        f.write('<h2>Trivy Vulnerabilities</h2>')
        f.write(f'<p>Threshold: <b>{summary.get("threshold")}</b></p>')
        if vulns:
            f.write('<table><thead><tr><th>ID</th><th>Package</th><th>Severity</th><th>Description</th></tr></thead><tbody>')
            for v in vulns:
                f.write(f'<tr><td>{html.escape(str(v.get("id") or ""))}</td>'
                        f'<td>{html.escape(str(v.get("package") or ""))}</td>'
                        f'<td>{html.escape(str(v.get("severity") or ""))}</td>'
                        f'<td><pre style="white-space:pre-wrap;margin:0;">{html.escape(v.get("description") or "")}</pre></td></tr>')
            f.write('</tbody></table>')
        else:
            f.write('<p>None</p>')

        # Sonar Errors section
        f.write('<h2>SonarQube Errors</h2>')
        if sonar_errors:
            f.write('<table><thead><tr><th>Rule</th><th>File:Line</th><th>Message</th></tr></thead><tbody>')
            for e in sonar_errors:
                f.write(f'<tr><td>{html.escape(str(e.get("rule") or ""))}</td>'
                        f'<td>{html.escape(str(e.get("file_line") or ""))}</td>'
                        f'<td><pre style="white-space:pre-wrap;margin:0;">{html.escape(str(e.get("message") or ""))}</pre></td></tr>')
            f.write('</tbody></table>')
        else:
            f.write('<p>None</p>')

        # GitGuardian Errors section
        f.write('<h2>GitGuardian Errors</h2>')
        if gg_errors:
            f.write('<table><thead><tr><th>Rule</th><th>File:Line</th><th>Message</th></tr></thead><tbody>')
            for e in gg_errors:
                f.write(f'<tr><td>{html.escape(str(e.get("rule") or ""))}</td>'
                        f'<td>{html.escape(str(e.get("file_line") or ""))}</td>'
                        f'<td><pre style="white-space:pre-wrap;margin:0;">{html.escape(str(e.get("message") or ""))}</pre></td></tr>')
            f.write('</tbody></table>')
        else:
            f.write('<p>None</p>')

        f.write('</body></html>')

# --- CLI ---
def main_cli():
    parser = argparse.ArgumentParser(description='Generate synthetic report from CycloneDX SBOM and SARIF')
    parser.add_argument('--sbom', required=False, help='Path to sbom-report.cdx.json (optional)')
    parser.add_argument('--sonar-sarif', required=True, help='Path to sonar-report.sarif')
    parser.add_argument('--gg-sarif', required=False, help='Path to gitguardian-report.sarif (optional)')
    parser.add_argument('--threshold', default='HIGH', help='Minimum severity to include (CRITICAL/HIGH/MEDIUM/LOW/INFO)')
    parser.add_argument('--outdir', default='/output', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    thr = args.threshold.upper()
    thr_v = sev_val(thr)

    sbom = load_json(args.sbom) if args.sbom else {}
    if not args.sbom:
        print('No SBOM provided; skipping SBOM-based vulnerability extraction')

    vulns = extract_vulnerabilities(sbom, thr_v)
    vulns = sorted(vulns, key=lambda v: (-sev_val(v.get('severity') or ''), v.get('package') or ''))

    sonar_errors = extract_sarif_errors(load_json(args.sonar_sarif))
    gg_errors = extract_sarif_errors(load_json(args.gg_sarif)) if args.gg_sarif else []

    counts = {'CRITICAL':0,'HIGH':0,'MEDIUM':0,'LOW':0,'INFO':0,'UNKNOWN':0}
    for v in vulns:
        s = (v.get('severity') or 'UNKNOWN').upper()
        counts[s] = counts.get(s,0)+1

    # Extract base image and OS info from SBOM
    try:
        meta = sbom.get('metadata',{}) or {}
        comp = meta.get('component',{}) or {}
        base_image = comp.get('name') or comp.get('purl') or comp.get('bom-ref') or ''
        for p in comp.get('properties',[]) or []:
            if p.get('name') in ('aquasecurity:trivy:RepoTag','aquasecurity:trivy:RepoDigest'):
                base_image = p.get('value') or base_image
                break
    except Exception:
        base_image = ''
    os_info = ''
    try:
        for c in sbom.get('components',[]) or []:
            if (c.get('type') or '').lower()=='operating-system':
                name = c.get('name') or ''
                ver = c.get('version') or ''
                os_info = f"{name} {ver}" if name and ver else name or ver or ''
                break
    except Exception:
        os_info = ''

    summary = {
        'threshold': thr,
        'counts': counts,
        'total_vulnerabilities': len(vulns),
        'sonar_errors': sonar_errors,
        'gg_errors': gg_errors,
        'base_image': base_image,
        'operating_system': os_info,
        'vulnerabilities': vulns
    }

    json_out = os.path.join(args.outdir,'synthetic-report.json')
    html_out = os.path.join(args.outdir,'synthetic-report.html')
    with open(json_out,'w') as f:
        json.dump(summary,f,indent=2)
    write_html(html_out, summary)
    print('Wrote:', json_out, html_out)

if __name__ == '__main__':
    main_cli()

