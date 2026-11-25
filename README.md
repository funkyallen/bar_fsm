# BAR-FSM

Research code for *Auditing Resource-Induced Loss of Goal Reachability in Tool-Using Agents* by Jianwen Rui and Shipei Liu.

BAR-FSM audits tool proposals that pass local checks yet can spend resources needed to finish a transaction. It compares the quota that would remain after a write with a lower bound on the outstanding work. Offline checks use sandbox state and native tool behavior to verify the transition.

## Run the synthetic checks

Use a Python 3.12 virtual environment. The release was verified with Python 3.12.14 on Windows.

```sh
python -m pip install -r requirements.txt
python -B bar_fsm.py demo
python -B bar_fsm.py qualify --out outputs/qualification.json
```

Both commands generate their fixtures locally and run without API credentials. The demo covers the free-void counterexample, a valid no-void deficit, and two installment choices with different successors. Qualification regenerates the 720 constructed accounting and service pairs. Under authoritative voidability, the certificate identifies 21 of 36 new losses, while the local gate would withhold the other 15 for insufficient evidence.

## Reproduce the paper

The full replay requires the companion `BAR-FSM_v5.2_IEEE_Access_20261125.zip` from the authors or the paper's supplementary materials. This code repository contains aggregate table values in [results](results), while the companion archive supplies the raw model replies, simulator states, request ledgers, and pinned Retail evaluator.

```sh
python -B bar_fsm.py reproduce --archive /path/to/BAR-FSM_v5.2_IEEE_Access_20261125.zip --out outputs/paper
```

The command verifies the archive and release source hashes, replays the study offline, and compares 14 regenerated tables and audit outputs with the archived references. Start with an empty output directory. Figure rendering may vary with platform and fonts.

## Code

- [runtime/scripts/research_v40_control.py](runtime/scripts/research_v40_control.py) implements checks at the tool boundary for the archived study.
- [runtime/scripts/research_accounting_void_contract.py](runtime/scripts/research_accounting_void_contract.py) implements the accounting bound with authoritative voidability.
- [runtime/src/aroute/envs](runtime/src/aroute/envs) contains the sandbox tool implementations.
- [audit_oracle_v50.py](audit_oracle_v50.py) verifies reachability using native witnesses and finite-state search.
- [audit_qualification_stages.py](audit_qualification_stages.py) separates holds at the local gate from unresolved resource checks.
- [reproduce_paper.py](reproduce_paper.py) regenerates the analyses, tables, and figures that support the manuscript.
- [results](results) contains aggregate values for the main workflow and family-level contrasts.

The runtime files retain the bytes recorded in the experiment protocols. The public entry point and the audit scripts use those files to run synthetic fixtures and replay sandbox traces.

## License and citation

The BAR-FSM code and repository documentation use the [MIT License](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) for the pinned benchmark used in the separate reproduction archive. Citation metadata for this software release is in [CITATION.cff](CITATION.cff).
