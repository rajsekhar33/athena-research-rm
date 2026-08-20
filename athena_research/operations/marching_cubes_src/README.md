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
