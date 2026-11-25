"""Narrow, recorded compatibility for the AASIST import-only repair.

Original run/model/token identities are retained only for exact reviewed code
changes and identical non-code inputs. Effective code gets a separate receipt;
historical RUN.json and completed model/cache files are never rewritten.
"""
from copy import deepcopy
from pathlib import Path

from .common import digest, read, require, sha

PATCH_ID = 'aasist_absolute_import_v1'
RECEIPT = 'AASIST_IMPORT_PATCH.json'
MANIFEST = Path(__file__).parent / 'patches' / (PATCH_ID + '.json')
PATCH_FILES = {
    'src/models/aasist.py',
    'scripts/trainable_or_3way/experiment.py',
    'scripts/trainable_or_3way/resume_compat.py',
}


def resume_description(previous, current):
    if previous == current:
        return current, None
    before = deepcopy(previous)
    after = deepcopy(current)
    old_code = before.pop('implementation')
    new_code = after.pop('implementation')
    require(before == after,
            'Existing run settings, data, assets or inputs changed; use a new output')
    patch = read(MANIFEST)
    require(patch.get('patch_id') == PATCH_ID and set(patch['files']) == PATCH_FILES,
            'Invalid AASIST import compatibility manifest')
    changed = {name for name in old_code.keys() | new_code.keys()
               if old_code.get(name) != new_code.get(name)}
    require(changed == PATCH_FILES,
            'Code changes are not exactly the reviewed AASIST import fix: ' + str(sorted(changed)))
    for name, pair in patch['files'].items():
        require(old_code.get(name) == pair['before'] and new_code.get(name) == pair['after'],
                'Unreviewed code revision in AASIST import fix: ' + name)
    receipt = dict(patch_id=PATCH_ID, patch_manifest=patch, patch_manifest_sha256=sha(MANIFEST),
                   original_run_identity=digest(previous),
                   effective_run_identity=digest(current),
                   effective_implementation=new_code,
                   policy='Import-only compatibility; original run, checkpoint and token identities retained; '
                          'all data, settings, assets and unrelated source files must match exactly')
    return deepcopy(previous), receipt
