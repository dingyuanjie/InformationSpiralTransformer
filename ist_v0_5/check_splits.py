"""Print and enforce the v0.5 strict split audit."""
import json
from strict_data import assert_no_leakage, split_audit

assert_no_leakage()
print(json.dumps(split_audit(), indent=2))
