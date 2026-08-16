# Level 7.0: evidence freeze and reproducibility bundle

Level 6.19.6 closed the router-repair branch. Level 7.0 does not train another
model or evaluate another candidate. It freezes the claims that survived the
registered experiments, preserves negative results, verifies their source JSON,
and builds a lightweight review bundle suitable for GitHub.

## Run

From `ist_v0_1`:

```powershell
python run_level7_0_local.py
```

The audit is CPU-only and normally finishes in a few seconds. Use `--force` to
replace an existing Level 7.0 output. Use `--no-zip` to run the complete audit
without creating the distributable archive.

## Outputs

The command writes:

- `formal/claim_audit.json`: every registered JSON-pointer check;
- `formal/reproducibility_manifest.json`: SHA-256 and size of every bundled
  source artifact;
- `formal/environment.json`: Python, PyTorch, CUDA, platform, and Git state;
- `formal/AUDIT_REPORT.md`: human-readable pass/fail report;
- `formal/ist_level7_0_repro_bundle.zip`: lightweight GitHub/reviewer archive;
- `formal/progress.json`: terminal status.

The ZIP excludes checkpoints, learned probe/router weights, per-example
predictions, partial outputs, smoke runs, and the failed Level 6.19.4 numerical
audit. Those remain in the local evidence archive and are represented by the
hash/provenance policy rather than duplicated.

## Boundary

This is a provenance and packaging pass, not a new efficacy result and not an
independent-initialization confirmation. A packaging failure authorizes fixing
only a missing path, malformed registry entry, hashing defect, or archive
defect. It does not authorize changing a scientific threshold or claim.

Seed909, protected tests, the Level 6.19.6 formal split, optimizer search, and
the closed router-repair branch remain locked.
