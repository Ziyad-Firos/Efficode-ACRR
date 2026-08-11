"""
ml/t5/ — training-pair generation for a possible CodeT5+ fine-tune.

See CODET5_PLAN.md (or CODEBERT_PLAN_V2.md's successor) in the repo root
for the full plan and the go/no-go gate this dataset feeds into. Nothing
in this package is used at request time — it exists to produce
pairs.jsonl for an offline fine-tuning step (plan Day 3), not to run in
the deployed app.
"""
