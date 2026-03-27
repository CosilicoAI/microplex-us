# microplex-us

US-specific survey adapters, calibration targets, pipelines, and PolicyEngine integration
built on top of the generic `microplex` engine.

## Docs

- [Docs index](./docs/README.md)
- [Architecture](./docs/architecture.md)
- [Source semantics](./docs/source-semantics.md)
- [Benchmarking](./docs/benchmarking.md)

## Current focus

`microplex-us` is being built as a library-first replacement path for
`policyengine-us-data`:

- canonical source and target metadata
- PE-US-compatible export
- full-target benchmarking against the active targets DB
- run registry and DuckDB index for frontier analysis

The architecture is still evolving, so the docs are deliberately technical and
operational rather than paper-like.
