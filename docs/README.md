# microplex-us docs

- [Architecture](./architecture.md)
- [Source semantics](./source-semantics.md)
- [Benchmarking](./benchmarking.md)
- [Methodology ledger](./methodology-ledger.md)
- [PolicyEngine oracle compatibility path](./policyengine-oracle-compatibility.md)
- [PE construction parity](./pe-construction-parity.md)
- [Superseding `policyengine-us-data`](./superseding-policyengine-us-data.md)

This doc set is intentionally technical. It is meant to answer six questions:

1. What is the current architecture?
2. How do source semantics and variable semantics drive donor integration?
3. Which construction contracts currently match PE, and which are only
   compatible?
4. How do we measure progress against `policyengine-us-data` on real targets?
5. What is the actual roadmap for fully superseding `policyengine-us-data`?
6. Which methodological choices are currently canonical, provisional, or open?

The docs describe the code that exists today. They do not try to freeze a final
paper narrative while the architecture is still moving.
