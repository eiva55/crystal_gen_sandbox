import torch
cached = torch.load('/tmp/magneli_placeholder_sym.pt')
for entry in cached:
    print(entry['mp_id'], '-> spacegroup:', entry.get('spacegroup', 'MISSING'))
