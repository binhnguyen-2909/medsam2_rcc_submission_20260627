# Tests

There is no automated unit-test suite in this submission snapshot.

Current validation is artifact- and run-level:

- inspect documented commands in `README.md` and `src/README.md`;
- create reproducible run folders with `scripts/create_run.py`;
- validate transferred artifacts with `scripts/export_checksums.py` and
  `scripts/validate_artifacts.py`;
- compare metrics against `results/tables/` and the summaries in
  `results/README.md`.

If code development continues, add focused tests here for path resolution,
box parsing/clipping, CSV batch inference, and checksum validation.
