# Local copy of AthenaK marching-cubes source

`marching_cubes.hpp` and `mc_luts.hpp` are a copy of
`athenak-RM/src/utils/{marching_cubes.hpp,mc_luts.hpp}` -- the shared C++
header the `area1`/`area2`/`area4` history diagnostic calls `process_cube`
from, and the same routine `athena_research.operations.area_functions`
compiles into a CUDA kernel for offline isosurface-area calculation.

Copied here rather than read from a home-relative `athenak-RM` checkout so
this package doesn't break or silently drift if that checkout moves, is on
a different machine, or is at a different revision than whatever produced
a given run's snapshots.

Licensed under the 3-clause BSD License, same as the rest of AthenaK
(`Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code
team`); the license header at the top of each file is unmodified.

**Not auto-synced.** If `process_cube`/the marching-cubes lookup tables are
ever changed upstream in `athenak-RM`, this copy must be refreshed by hand
for the area calculation to keep matching the pgen's method exactly.

**Local deviation:** `interior_vertex()` (tiling entry 12) falls back to the
`process_cube` classification (`v < 0` is inside) when no cube edge has a
strict sign change, i.e. when three or more corners lie exactly on the iso
level. Upstream divides 0/0 there and returns NaN; every finite result is
unchanged. `area_functions._ADD_TRIANGLE_PHYS` carries the same rule as
`mc_interior_vertex()` so it compiles against any copy of this header.
