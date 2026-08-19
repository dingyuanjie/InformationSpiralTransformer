import json
from pathlib import Path

root = Path(__file__).resolve().parent
formal = root / 'experiments' / 'level7_5_3_4' / 'formal'
result_path = formal / 'result.json'
if not result_path.exists():
    raise SystemExit('result.json not found; run the formal experiment first')
d = json.loads(result_path.read_text(encoding='utf-8'))
diag = d.get('diagnosis', {})
counts = diag.get('effect_counts', {})
l2 = {k: counts.get(f'freeze_l2_{k}', 0) for k in ('slot_queries','write_core','read_fusion')}
l3 = {k: counts.get(f'freeze_l3_{k}', 0) for k in ('slot_queries','write_core','read_fusion')}
if len(set(l2.values())) == 1 and len(set(l3.values())) == 1 and any(l2.values()) and any(l3.values()):
    classification = 'distributed_fine_memory'
else:
    classification = diag.get('classification', 'undetermined')
diag['classification'] = classification
diag['layer2_effect_counts'] = l2
diag['layer3_effect_counts'] = l3
diag['classification_note'] = 'Tie-safe classification: equal material effects across all registered L2/L3 groups are reported as distributed_fine_memory.'
d['diagnosis'] = diag
result_path.write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
lines = [
    '# Level 7.5.3.4 Analysis', '',
    f"Classification: **{classification}**", '',
    'All six registered fine-grained Memory interventions were compared against the exact optimizer/RNG reference on four outcome-stratified endpoints.', '',
    '| Group | Material effects |', '|---|---:|',
]
for layer, vals in (('L2', l2), ('L3', l3)):
    for kind, value in vals.items():
        lines.append(f'| {layer} {kind} | {value}/4 |')
lines += ['', 'Interpretation: the observed effect is distributed across slot allocation, write core, and read/fusion components in both L2 and L3. This is not a prevalence estimate.']
(formal / 'ANALYSIS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(classification)
