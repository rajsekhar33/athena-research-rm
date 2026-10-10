"""Periodic self-neighbor lookup and affected gradient callers."""
from types import SimpleNamespace

import numpy as np
import pytest

from athena_research.core.base import asnumpy
from athena_research.operations.grad_div_curl import _gradient_mb
from athena_research.utils.meshblock_utils import find_neighbour_blocks
from boundary_mesh import BoundaryMesh


@pytest.mark.parametrize('axis', ['x','y','z'])
@pytest.mark.parametrize('sign', ['+','-'])
def test_one_block_periodic_direction_returns_self(axis, sign):
    mesh = BoundaryMesh(np.zeros((4,4,4)), periodic=(True,True,True))
    assert find_neighbour_blocks(mesh, 0, axis+sign) == [0]


@pytest.mark.parametrize('axis', ['x','y','z'])
@pytest.mark.parametrize('sign', ['+','-'])
def test_one_block_nonperiodic_direction_has_no_neighbor(axis, sign):
    mesh = BoundaryMesh(np.zeros((4,4,4)), periodic=(False,False,False))
    assert find_neighbour_blocks(mesh, 0, axis+sign) == []


@pytest.mark.parametrize('axis', ['x','y','z'])
def test_same_level_multiblock_wrap_does_not_return_self(axis):
    mesh = BoundaryMesh(np.zeros((8,8,8)), periodic=(True,True,True))
    for m,logical in enumerate(mesh.mb_logical):
        target = logical.copy()
        target['xyz'.index(axis)] = 1-target['xyz'.index(axis)]
        expected = np.flatnonzero(np.all(mesh.mb_logical == target,axis=1)).tolist()
        for sign in ('+','-'):
            assert find_neighbour_blocks(mesh, m, axis+sign) == expected
            assert m not in expected


def test_periodic_single_root_refined_direction_is_not_self():
    mesh = BoundaryMesh(np.zeros((8,8,8)), periodic=(True,True,True))
    # Eight level-1 leaves within a single root block.
    mesh.Nx1 = mesh.Nx2 = mesh.Nx3 = 4
    mesh.mb_logical[:,3] = 1
    assert find_neighbour_blocks(mesh, 0, 'x+') == [1]
    assert find_neighbour_blocks(mesh, 1, 'x+') == [0]


def test_fine_to_coarse_neighbor_lookup_is_unchanged():
    mesh = SimpleNamespace(n_mbs=5, Nx1=8, Nx2=4, Nx3=4,
                           nx1=4,nx2=4,nx3=4,
                           mb_logical=np.array([(0,0,0,0), (2,0,0,1),
                                                (2,1,0,1), (2,0,1,1),
                                                (2,1,1,1)]),
                           _header={'mesh': {f'ix{i}_bc':'outflow' for i in (1,2,3)}})
    for m in (1,2,3,4):
        assert find_neighbour_blocks(mesh, m, 'x-') == [0]


@pytest.mark.parametrize('axis', ['x','y','z'])
def test_single_block_periodic_gradient_matches_centered_wrap(axis):
    mesh = BoundaryMesh(np.zeros((8,8,8)), block_shape=(8,8,8),
                        periodic=(True,True,True))
    array_axis = 2-'xyz'.index(axis)
    coordinate = np.arange(8)+.5
    shape = [1,1,1]
    shape[array_axis] = 8
    field = np.broadcast_to(np.sin(2*np.pi*coordinate/8).reshape(shape),
                            (8,8,8)).copy()
    mesh.fields['wave'] = field[np.newaxis]
    expected = (np.roll(field,-1,axis=array_axis)-np.roll(field,1,axis=array_axis))/(
        2*mesh.spacing['xyz'.index(axis)])
    actual = _gradient_mb(mesh,'wave',0,1,axis=axis,handle_boundaries=True)
    np.testing.assert_allclose(asnumpy(actual)[0],expected,rtol=1e-12,atol=1e-12)
