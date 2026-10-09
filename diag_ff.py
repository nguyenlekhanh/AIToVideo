import json

d = json.load(open('workflows_template/video_minimax_h3_t2v.json'))
sub = next(s for s in d['definitions']['subgraphs']
           if 'MiniMax' in s.get('name', ''))
inner = {n['id']: n for n in sub['nodes']}
n131 = inner[131]
print('node 131 type:', n131['type'])
print('131 inputs (slot order):')
for i, inp in enumerate(n131.get('inputs', [])):
    print('  slot %d: name=%s type=%s link=%s widget=%s' % (
        i, inp.get('name'), inp.get('type'), inp.get('link'),
        (inp.get('widget') or {}).get('name') if inp.get('widget') else None))
print('131 widgets count:', len(n131.get('widgets_values') or []))
print()
print('subgraph inputs feeding frames:')
for i, inp in enumerate(sub.get('inputs', [])):
    if 'frame' in (inp.get('name') or '').lower():
        print('  slot %d: %s linkIds=%s' % (i, inp.get('name'), inp.get('linkIds')))
print()
print('links into 131 slots 2,3 (first/last frame):')
for L in sub['links']:
    if L['target_id'] == 131 and L['target_slot'] in (2, 3):
        print('  link %s: origin=%s slot=%s' % (L['id'], L['origin_id'], L['origin_slot']))
print()
print('INSTANCE 140 frame slots:')
inst = next(n for n in d['nodes'] if n['id'] == 140)
for i, inp in enumerate(inst.get('inputs', [])):
    if 'frame' in (inp.get('name') or '').lower():
        print('  slot %d: %s link=%s' % (i, inp.get('name'), inp.get('link')))
