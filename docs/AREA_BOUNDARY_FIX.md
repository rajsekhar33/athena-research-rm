# Shared meshblock boundary fixes

## Changes

- `calc_area_mb_based` now constructs the bounded padded block before its
  min/max culling check. Face, edge and corner vertices therefore participate
  in the check, retaining interfaces between uniform neighboring blocks.
  Only a padded block that straddles the requested level goes to the GPU.
  Exact-level vertices are warm, matching the unchanged cube kernel's
  strict-negative classification.
- `find_neighbour_blocks` no longer unconditionally excludes the source
  block. It can match itself after periodic wrapping in a single-block
  direction. Ordinary nonperiodic, multiblock and refined same-level targets
  remain distinct from the source.
- The old snapshot-dependent regression expecting zero area at a meshblock
  face now expects a finite positive result. Its field accesses use explicit
  meshblock bounds.

This is a small adaptation to shared infrastructure, not a wholesale port
of problem-specific physics, drivers, research data or cache-repair tools.
No core, CUDA kernel, temperature target, slab-selection policy, physical
normalization or cache invalidation behavior changes.

## Tests

`tests/test_operations_area_boundaries.py` compiles the exact bundled
`process_cube_phys` C++ geometry using g++ for CPU orchestration tests, then
also exercises the actual CUDA kernel when available. The independent
reference is an unculled globally padded synthetic field. Tests cover:

- Analytic meshblock-face planes in all three directions, corner-only
  crossings and exact-level vertices.
- Random fields over five meshblock partitions, constant fields, and a
  single-block periodic z direction.
- The real interface-slab selection for a boundary-aligned plane.
- Three-rank MPI work sharing/allreduce with replicated mesh data.

`tests/test_meshblock_periodic_neighbors.py` covers both signs of each
periodic/nonperiodic direction, multiblock wrapping, refined single-root
layouts, fine-to-coarse lookup, and all three single-block periodic
gradients against independently centered wrapped differences.

Validation on 2026-10-10, based on public main commit `f95b9d47`:

| Lane | Result |
|---|---|
| Complete CPU suite | 72 passed, 46 skipped |
| Complete CUDA suite | 90 passed, 28 skipped |
| Three-rank CPU area/SF checks | 4 passed, 1 skipped per rank |
| Three-rank CUDA area/SF checks | 5 passed per rank |
| Six changed/new Python files | Compile passed |
| Diff whitespace | Passed |

Snapshot-dependent checks are skipped without test data. CUDA tests skip
under CPU-only mode, MPI checks skip in a single-rank invocation, and CPU
geometry tests require g++. Both complete-suite runs emit the existing
optional `units/length_cgs` header warning. CUDA tests used a 512 MiB pool
limit; three MPI GPU ranks shared one GPU. These are synthetic correctness
checks, not production performance or separate-device scaling benchmarks.

Reproduce the suites from the checkout:

```bash
ATHENA_RESEARCH_CPU_ONLY=1 python -m pytest -q
ATHENA_RESEARCH_CPU_ONLY=0 python -m pytest -q
ATHENA_RESEARCH_CPU_ONLY=1 mpirun -n 3 python -m pytest -q \
  tests/test_operations_area_boundaries.py tests/test_mpi_shared_port.py \
  -k 'mpi_area or distributed'
ATHENA_RESEARCH_CPU_ONLY=0 mpirun -n 3 python -m pytest -q \
  tests/test_operations_area_boundaries.py tests/test_mpi_shared_port.py \
  -k 'mpi_area or distributed'
```

Use a scratch `--basetemp` for generated test artifacts; use a separate
directory per MPI rank if specifying it explicitly.

## Boundaries and remaining issues

- Shared meshblock area padding still assumes uniform blocks, periodic
  x1/x2, and outflow x3 (periodic z also tested). This patch is not an AMR
  triangulation or arbitrary-boundary-condition implementation.
- MPI area tests use a complete tiny mesh on every rank. They validate
  disjoint block scheduling and summation, not distributed neighbor-halo
  exchange. The existing sharded-data limitations are not removed.
- Building padding before culling can fetch more neighbor data for inactive
  blocks; memory remains bounded, but a production timing benefit is not
  claimed.
- A separate pre-existing coarse-to-fine lookup issue was found during the
  audit: in a mixed-level nonperiodic mesh, the coarse block at logical
  (0,0,0,0) fails to find the four +x fine neighbors at x=2, level=1.
  The unmodified baseline and patched helper both return an empty list;
  fine-to-coarse lookup returns the correct coarse neighbor in both.
  This distinct AMR issue is documented, not repaired in this patch.
- Coarse-corner sampling can still lose thin interfaces. Boundary-safe
  culling does not turn coarse area-decay slopes into validated fractal
  dimensions.

Publication scope is limited to these generic source, test and documentation
changes; problem-specific files and generated artifacts are excluded.
