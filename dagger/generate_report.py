import json
import os
import html
import argparse
from typing import Dict, Any, List


def sev_val(s: str) -> int:
    m = {
        'CRITICAL': 5,
        'HIGH': 4,
        'MEDIUM': 3,
        'LOW': 2,
        'INFO': 1,
        'UNKNOWN': 0,
    }
    return m.get((s or '').upper(), 0)


def load_json(path: str):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}


def extract_vulnerabilities(sbom: dict, thr_v: int):
    vulns: List[Dict[str, Any]] = []
    if not isinstance(sbom, dict):
        return vulns

    # build a map of components by bom-ref for resolving 'affects' refs
    components = {}
    for c in sbom.get('components', []) or []:
        ref = c.get('bom-ref')
        if ref:
            components[ref] = c

    # Top-level vulnerabilities array (some CycloneDX outputs)
    if 'vulnerabilities' in sbom and isinstance(sbom['vulnerabilities'], list):
        for v in sbom['vulnerabilities']:
            # prefer CVSSv3/3.1 ratings when present
            sev = 'UNKNOWN'
            ratings = v.get('ratings') or []

            # 1) CVSSv3 preferred
            for r in ratings:
                method = (r.get('method') or '').upper()
                if 'CVSSV3' in method:
                    sev = (r.get('severity') or '').upper() or sev
                    break

            # 2) fallback to NVD source rating
            if sev == 'UNKNOWN':
                for r in ratings:
                    src = r.get('source') or {}
                    if isinstance(src, dict) and src.get('name', '').lower() == 'nvd':
                        sev = (r.get('severity') or '').upper() or sev
                        break

            # 3) fallback to best (highest) rating among all ratings
            if sev == 'UNKNOWN' and isinstance(ratings, list) and ratings:
                best = max((sev_val(r.get('severity') or '') for r in ratings), default=0)
                for name, val in [('CRITICAL', 5), ('HIGH', 4), ('MEDIUM', 3), ('LOW', 2), ('INFO', 1)]:
                    if val == best:
                        sev = name
                        break

            # 4) final fallback to explicit fields
            if sev == 'UNKNOWN':
                sev = (v.get('severity') or v.get('severityName') or 'UNKNOWN').upper()

            if sev_val(sev) >= thr_v:
                vid = v.get('id') or ''
                desc = v.get('description') or v.get('detail') or ''

                # 1) prefer explicit 'affects' ref -> resolve to component name
                pkg = ''
                affects = v.get('affects') or []
                if isinstance(affects, list) and affects:
                    ref = affects[0].get('ref') if isinstance(affects[0], dict) else affects[0]
                    if isinstance(ref, str):
                        # if ref points to a component bom-ref, use that component's name
                        comp = components.get(ref)
                        if comp:
                            pkg = comp.get('name') or comp.get('purl') or ref
                        else:
                            # if ref is a purl, try to parse a human name
                            if ref.startswith('pkg:'):
                                try:
                                    last = ref.split('/')[-1]
                                    pkg = last.split('@')[0]
                                except Exception:
                                    pkg = ref
                            else:
                                pkg = ref

                # 2) fall back to common fields if not found
                if not pkg:
                    raw_pkg = v.get('component') or v.get('module') or v.get('package') or ''
                    if isinstance(raw_pkg, dict):
                        pkg = raw_pkg.get('name') or raw_pkg.get('bom-ref') or str(raw_pkg)
                    else:
                        # avoid using vulnerability id as package if possible
                        if raw_pkg and raw_pkg != vid and not (isinstance(raw_pkg, str) and (raw_pkg.lower().startswith('cve') or raw_pkg.lower().startswith('vuln'))):
                            pkg = raw_pkg
                        else:
                            pkg = v.get('module') or v.get('package') or v.get('componentName') or v.get('component_name') or vid

                vulns.append({'id': vid, 'package': pkg, 'severity': sev, 'description': desc})
        return vulns

    # Component-based vulnerabilities
    for c in sbom.get('components', []):
        for v in c.get('vulnerabilities', []):
            sev = (v.get('severity') or 'UNKNOWN').upper()
            if sev_val(sev) >= thr_v:
                pkg = c.get('name') or c.get('bom-ref') or c.get('id')
                desc = v.get('description', '')
                vulns.append({'package': pkg, 'severity': sev, 'description': desc, 'id': v.get('id')})

    return vulns


def extract_sarif_errors(sarif: dict):
    errors: List[Dict[str, Any]] = []
    if not isinstance(sarif, dict):
        return errors

    for run in sarif.get('runs', []):
        # build map of ruleId -> defaultConfiguration.level
        rule_level = {}
        driver = run.get('tool', {}).get('driver', {})
        for rule in driver.get('rules', []) or []:
            rid = rule.get('id')
            lvl = (rule.get('defaultConfiguration', {}).get('level') or '').lower()
            if rid:
                rule_level[rid] = lvl

        for res in run.get('results', []):
            # result level may be omitted; fall back to rule default
            level = (res.get('level') or '').lower()
            if not level:
                level = rule_level.get(res.get('ruleId'), '')

            if level == 'error':
                msg = res.get('message', {})
                text = msg.get('text') if isinstance(msg, dict) else str(msg)
                # extract file and line from first location if available
                file_line = ''
                locs = res.get('locations', []) or []
                if locs and isinstance(locs, list):
                    try:
                        pl = locs[0].get('physicalLocation', {})
                        art = pl.get('artifactLocation', {})
                        uri = art.get('uri') or art.get('uri', '')
                        region = pl.get('region', {})
                        start = region.get('startLine') or region.get('startColumn')
                        if uri:
                            if start:
                                file_line = f"{uri}:{start}"
                            else:
                                file_line = uri
                    except Exception:
                        file_line = ''

                errors.append({'rule': res.get('ruleId'), 'message': text, 'locations': locs, 'file_line': file_line})

    return errors


def write_html(path: str, summary: Dict[str, Any]):
    vulns = summary.get('vulnerabilities', [])
    errors = summary.get('errors', [])
    counts = summary.get('counts', {})

    with open(path, 'w', encoding='utf-8') as f:
        f.write('<!doctype html><html><head><meta charset="utf-8"><title>Synthetic Report</title>')
        f.write('<style>table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:8px}th{background:#f2f2f2;text-align:left}</style>')
        f.write('</head><body>')
        f.write('<h1>Synthetic Report</h1>')
        # show base image and operating system (if available) before Trivy vulnerabilities
        base_img = summary.get('base_image')
        os_info = summary.get('operating_system')
        if base_img or os_info:
            f.write('<h2>Environment</h2>')
            f.write('<table><tbody>')
            if base_img:
                f.write('<tr><th style="text-align:left;padding:8px">Base image</th><td style="padding:8px">{}</td></tr>'.format(html.escape(str(base_img))))
            if os_info:
                f.write('<tr><th style="text-align:left;padding:8px">Operating system</th><td style="padding:8px">{}</td></tr>'.format(html.escape(str(os_info))))
            f.write('</tbody></table>')

        # Counts as table (hide zero entries)
        f.write('<h2>Counts</h2>')
        f.write('<table><thead><tr><th style="text-align:left;padding:8px">Metric</th><th style="text-align:left;padding:8px">Count</th></tr></thead><tbody>')
        for k in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO', 'UNKNOWN']:
            c = counts.get(k, 0)
            if c > 0:
                f.write('<tr><td style="padding:8px">VULNERABILITIES - {}</td><td style="padding:8px">{}</td></tr>'.format(html.escape(str(k)), c))
        # Sonar error-level issues count
        sonar_err_count = summary.get('errors_count') or 0
        if sonar_err_count:
            f.write('<tr><td style="padding:8px">SOURCE CODE - ERRORS</td><td style="padding:8px">{}</td></tr>'.format(sonar_err_count))
        f.write('</tbody></table>')

        f.write('<h2>Trivy Vulnerabilities</h2>')
        f.write(f'<p>Threshold: <b>{summary.get("threshold")}</b></p>')

        if vulns:
            f.write('<table><thead><tr><th>ID</th><th>Package</th><th>Severity</th><th>Description</th></tr></thead><tbody>')
            for v in vulns:
                vid = (v.get('id') or '')
                pkg = (v.get('package') or '')
                sev = (v.get('severity') or '')
                # preserve original line breaks and escape HTML
                desc = html.escape(v.get('description') or '')
                f.write('<tr>')
                f.write('<td>{}</td>'.format(html.escape(str(vid))))
                f.write('<td>{}</td>'.format(html.escape(str(pkg))))
                f.write('<td>{}</td>'.format(html.escape(str(sev))))
                f.write('<td><pre style="white-space:pre-wrap;margin:0;">{}</pre></td>'.format(desc))
                f.write('</tr>')
            f.write('</tbody></table>')
        else:
            f.write('<p>None</p>')

        f.write('<h2>SonarQube Errors</h2>')
        if errors:
            f.write('<table><thead><tr><th>Rule</th><th>File:Line</th><th>Message</th></tr></thead><tbody>')
            for e in errors:
                rule = (e.get('rule') or '')
                file_line = (e.get('file_line') or '')
                msg = html.escape(e.get('message') or '')
                f.write('<tr>')
                f.write('<td>{}</td>'.format(html.escape(str(rule))))
                f.write('<td>{}</td>'.format(html.escape(str(file_line))))
                f.write('<td><pre style="white-space:pre-wrap;margin:0;">{}</pre></td>'.format(msg))
                f.write('</tr>')
            f.write('</tbody></table>')
        else:
            f.write('<p>None</p>')

        f.write('</body></html>')


def main_cli():
    parser = argparse.ArgumentParser(description='Generate synthetic report from CycloneDX SBOM and SARIF')
    parser.add_argument('--sbom', required=True, help='Path to sbom-report.cdx.json')
    parser.add_argument('--sarif', required=True, help='Path to sonar-report.sarif')
    parser.add_argument('--threshold', default='HIGH', help='Minimum severity to include (CRITICAL/HIGH/MEDIUM/LOW/INFO')
    parser.add_argument('--outdir', default='/output', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    thr = args.threshold.upper()
    thr_v = sev_val(thr)

    sbom = load_json(args.sbom)
    sarif = load_json(args.sarif)

    # extract base image info from SBOM metadata.component if present
    base_image = ''
    try:
        meta = sbom.get('metadata', {}) or {}
        comp = meta.get('component', {}) or {}
        base_image = comp.get('name') or comp.get('purl') or comp.get('bom-ref') or ''
        # check properties for aquasecurity tag (RepoTag/RepoDigest)
        for p in comp.get('properties', []) or []:
            n = p.get('name', '')
            if n in ('aquasecurity:trivy:RepoTag', 'aquasecurity:trivy:RepoDigest'):
                base_image = p.get('value') or base_image
                break
    except Exception:
        base_image = ''

    # extract operating system from components (type == 'operating-system')
    os_info = ''
    try:
        for c in sbom.get('components', []) or []:
            if (c.get('type') or '').lower() == 'operating-system':
                name = c.get('name') or ''
                ver = c.get('version') or ''
                if name and ver:
                    os_info = f"{name} {ver}"
                else:
                    os_info = name or ver or ''
                break
    except Exception:
        os_info = ''

    vulns = extract_vulnerabilities(sbom, thr_v)
    errors = extract_sarif_errors(sarif)

    # Sort vulnerabilities from most critical to less critical
    vulns = sorted(vulns, key=lambda v: (-sev_val((v.get('severity') or '')), (v.get('package') or '') ))

    # Build per-severity counts so 'HIGH' threshold also shows CRITICAL + HIGH counts
    counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'INFO': 0, 'UNKNOWN': 0}
    for v in vulns:
        s = (v.get('severity') or 'UNKNOWN').upper()
        if s not in counts:
            counts['UNKNOWN'] += 1
        else:
            counts[s] = counts.get(s, 0) + 1

    summary = {
        'threshold': thr,
        'counts': counts,
        'total_vulnerabilities': len(vulns),
        'errors_count': len(errors),
        'base_image': base_image,
        'operating_system': os_info,
        'vulnerabilities': vulns,
        'errors': errors,
    }

    json_out = os.path.join(args.outdir, 'synthetic-report.json')
    html_out = os.path.join(args.outdir, 'synthetic-report.html')

    with open(json_out, 'w') as f:
        json.dump(summary, f, indent=2)

    write_html(html_out, summary)

    print('Wrote:', json_out, html_out)


if __name__ == '__main__':
    # support running as a script in Dagger containers
    main_cli()
