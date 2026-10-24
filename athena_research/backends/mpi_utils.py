"""
MPI utilities for distributed computing across multiple nodes and GPUs.
"""
from typing import Optional, Any, List, Tuple
import numpy as np

_COMM_OVERRIDE = None


def set_comm_override(comm):
    """Override the communicator used by MPIManager for controlled test runs."""
    global _COMM_OVERRIDE
    _COMM_OVERRIDE = comm


def clear_comm_override():
    """Clear any communicator override and fall back to MPI.COMM_WORLD."""
    global _COMM_OVERRIDE
    _COMM_OVERRIDE = None


class MPIManager:
    """
    Manager for MPI-based distributed computing.
    
    This class handles MPI initialization, communication, and coordination
    for distributed computation across multiple nodes/GPUs.
    """
    
    _init_message_printed = {}  # Track which ranks have printed init message
    
    def __init__(self, backend_manager=None, verbose=False):
        """
        Initialize MPI manager.
        
        Parameters
        ----------
        backend_manager : BackendManager, optional
            Backend manager instance for GPU/CPU coordination
        verbose : bool, optional
            If True, print initialization message. Default is False.
        """
        self.comm = None
        self.rank = 0
        self.size = 1
        self.is_initialized = False
        self.backend_manager = backend_manager
        
        try:
            from mpi4py import MPI
            self.comm = _COMM_OVERRIDE if _COMM_OVERRIDE is not None else MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
            self.is_initialized = True
            self.MPI = MPI
            
            # Only print once per rank (use class variable to track)
            if verbose and self.rank not in MPIManager._init_message_printed:
                print(f"MPI initialized: rank {self.rank}/{self.size}")
                MPIManager._init_message_printed[self.rank] = True
        except ImportError:
            if verbose:
                print("Warning: mpi4py not available. Running in single-process mode.")
    
    def barrier(self):
        """Synchronize all MPI processes."""
        if self.is_initialized:
            self.comm.Barrier()
    
    def broadcast(self, data, root=0):
        """
        Broadcast data from root process to all processes.
        
        Parameters
        ----------
        data : any
            Data to broadcast (only significant at root)
        root : int, optional
            Rank of root process
        
        Returns
        -------
        any
            Broadcasted data
        """
        if self.is_initialized:
            return self.comm.bcast(data, root=root)
        return data
    
    def scatter(self, data, root=0):
        """
        Scatter data from root to all processes.
        
        Parameters
        ----------
        data : list or None
            List of data chunks to scatter (only significant at root)
        root : int, optional
            Rank of root process
        
        Returns
        -------
        any
            Data chunk for this process
        """
        if self.is_initialized:
            return self.comm.scatter(data, root=root)
        return data[0] if data else None
    
    def gather(self, data, root=0):
        """
        Gather data from all processes to root.
        
        Parameters
        ----------
        data : any
            Data from this process
        root : int, optional
            Rank of root process
        
        Returns
        -------
        list or None
            List of data from all processes (only at root)
        """
        if self.is_initialized:
            return self.comm.gather(data, root=root)
        return [data]
    
    def allgather(self, data):
        """
        Gather data from all processes and distribute to all.
        
        Parameters
        ----------
        data : any
            Data from this process
        
        Returns
        -------
        list
            List of data from all processes
        """
        if self.is_initialized:
            return self.comm.allgather(data)
        return [data]
    
    _OP_NAMES = ('sum', 'prod', 'max', 'min')

    def _mpi_op(self, op):
        if op not in self._OP_NAMES:
            raise ValueError(f"unknown reduction op {op!r}; expected one of "
                             f"{self._OP_NAMES}")
        return {'sum': self.MPI.SUM, 'prod': self.MPI.PROD,
                'max': self.MPI.MAX, 'min': self.MPI.MIN}[op]

    def _reduce_impl(self, data, op, root):
        """Reduce ``data`` over all ranks; ``root=None`` means allreduce.

        Arrays and NumPy scalars are reduced elementwise. The pickle-based
        mpi4py reductions apply MIN/MAX with Python's ``min``/``max``, which
        raise on multi-element arrays on one rank while the others wait
        forever, so arrays go through the buffer interface (numeric dtypes)
        or an allgather plus the matching NumPy ufunc (anything else).
        Ranks may pass different numeric types for the same quantity (a Python
        float on a rank without meshblocks, np.float64 elsewhere): every rank
        first shares its kind, dtype and shape, so all of them take the same
        path and reduce in the common dtype. Mismatched shapes, or arrays on
        some ranks and other objects on others, raise on every rank.
        """
        mpi_op = self._mpi_op(op)
        to_device = None
        if self.backend_manager and self.backend_manager.backend_type == 'gpu':
            host = self.backend_manager.asnumpy(data)
            to_device = self.backend_manager.to_device
        elif type(data).__module__.split('.')[0] == 'cupy':
            import cupy
            host, to_device = cupy.asnumpy(data), cupy.asarray
        else:
            host = data
        numeric = isinstance(host, (np.ndarray, np.generic, bool, int, float, complex))
        if numeric:
            arr = np.array(host, order='C', copy=True)
            meta = ('array', arr.dtype.str, arr.shape)
        else:
            meta = ('object', None, None)
        metas = self.comm.allgather(meta)
        kinds = {m[0] for m in metas}
        if len(kinds) > 1:
            raise ValueError(f'MPI reduce: ranks passed incompatible values {metas}')
        if numeric:
            if len({m[2] for m in metas}) > 1:
                raise ValueError(f'MPI reduce: ranks passed different shapes {[m[2] for m in metas]}')
            arr = arr.astype(np.result_type(*[np.dtype(m[1]) for m in metas]), copy=False)
            is_scalar = arr.ndim == 0
            # MPI has no MIN/MAX for complex and no float16 datatype.
            buffer_ok = (arr.dtype.kind in 'iu'
                         or (arr.dtype.kind == 'f' and arr.dtype.itemsize >= 4)
                         or (arr.dtype.kind == 'c' and op in ('sum', 'prod')))
            if buffer_ok:
                flat = arr.reshape(-1)
                out = np.empty_like(flat)
                if root is None:
                    self.comm.Allreduce(flat, out, op=mpi_op)
                else:
                    self.comm.Reduce(flat, out if self.rank == root else None,
                                     op=mpi_op, root=root)
                    if self.rank != root:
                        return None
                result = out.reshape(arr.shape)
            else:
                ufunc = {'sum': np.add, 'prod': np.multiply,
                         'max': np.maximum, 'min': np.minimum}[op]
                if root is None:
                    parts = self.comm.allgather(arr)
                else:
                    parts = self.comm.gather(arr, root=root)
                    if self.rank != root:
                        return None
                result = ufunc.reduce(np.stack(parts), axis=0)
            if is_scalar:
                result = result[()]
            return to_device(result) if to_device is not None else result
        if root is None:
            result = self.comm.allreduce(host, op=mpi_op)
        else:
            result = self.comm.reduce(host, op=mpi_op, root=root)
            if self.rank != root:
                return None
        return to_device(result) if to_device is not None else result

    def reduce(self, data, op='sum', root=0):
        """
        Reduce data from all processes using specified operation.
        
        Parameters
        ----------
        data : array-like or scalar
            Data to reduce
        op : str, optional
            Reduction operation ('sum', 'prod', 'max', 'min')
        root : int, optional
            Rank of root process
        
        Returns
        -------
        array-like or scalar
            Reduced result (only at root; None elsewhere)
        """
        if not self.is_initialized:
            return data
        return self._reduce_impl(data, op, root)
    
    def allreduce(self, data, op='sum'):
        """
        Reduce data from all processes and distribute result to all.
        
        Parameters
        ----------
        data : array-like or scalar
            Data to reduce
        op : str, optional
            Reduction operation ('sum', 'prod', 'max', 'min')
        
        Returns
        -------
        array-like or scalar
            Reduced result at all processes
        """
        if not self.is_initialized:
            return data
        return self._reduce_impl(data, op, None)
    
    def send(self, data, dest, tag=0):
        """
        Send data to another process.
        
        Parameters
        ----------
        data : any
            Data to send
        dest : int
            Rank of destination process
        tag : int, optional
            Message tag
        """
        if self.is_initialized:
            # Convert GPU arrays to CPU before sending
            if self.backend_manager and self.backend_manager.backend_type == 'gpu':
                data = self.backend_manager.asnumpy(data)
            self.comm.send(data, dest=dest, tag=tag)
    
    def recv(self, source, tag=0):
        """
        Receive data from another process.
        
        Parameters
        ----------
        source : int
            Rank of source process
        tag : int, optional
            Message tag
        
        Returns
        -------
        any
            Received data
        """
        if self.is_initialized:
            data = self.comm.recv(source=source, tag=tag)
            # Convert back to GPU if needed
            if self.backend_manager and self.backend_manager.backend_type == 'gpu':
                if isinstance(data, np.ndarray):
                    data = self.backend_manager.to_device(data)
            return data
        return None

    def _workload_bounds(self, total_items: int, rank: Optional[int] = None) -> Tuple[int, int]:
        """Return the [start, end) workload bounds for a given rank."""
        if rank is None:
            rank = self.rank

        items_per_proc = total_items // self.size
        remainder = total_items % self.size

        if rank < remainder:
            start_idx = rank * (items_per_proc + 1)
            end_idx = start_idx + items_per_proc + 1
        else:
            start_idx = rank * items_per_proc + remainder
            end_idx = start_idx + items_per_proc

        return start_idx, end_idx

    def distribute_workload(self, total_items: int) -> Tuple[int, int]:
        """
        Distribute workload evenly across MPI processes.
        
        Parameters
        ----------
        total_items : int
            Total number of items to distribute
        
        Returns
        -------
        start_idx : int
            Starting index for this process
        end_idx : int
            Ending index (exclusive) for this process
        """
        return self._workload_bounds(total_items, rank=self.rank)
    
    def distribute_array(self, array, axis=0, root=0):
        """
        Distribute an array across MPI processes along a specified axis.
        
        Parameters
        ----------
        array : array-like or None
            Array to distribute (only significant at root)
        axis : int, optional
            Axis along which to split the array
        root : int, optional
            Rank of root process
        
        Returns
        -------
        array-like
            Local chunk for this process
        """
        if not self.is_initialized:
            return array
        
        # Broadcast shape information
        if self.rank == root:
            shape = array.shape
            dtype = array.dtype
        else:
            shape = None
            dtype = None
        
        shape = self.broadcast(shape, root=root)
        dtype = self.broadcast(dtype, root=root)
        
        split_size = shape[axis] // self.size
        remainder = shape[axis] % self.size

        if self.rank == root:
            chunks = []
            start = 0
            for i in range(self.size):
                chunk_size = split_size + (1 if i < remainder else 0)
                end = start + chunk_size
                
                # Get slice for this chunk
                slices = [slice(None)] * len(shape)
                slices[axis] = slice(start, end)
                
                # Convert to CPU if needed
                chunk = array[tuple(slices)]
                if self.backend_manager and self.backend_manager.backend_type == 'gpu':
                    chunk = self.backend_manager.asnumpy(chunk)
                
                chunks.append(chunk)
                start = end
        else:
            chunks = None
        
        local_chunk = self.scatter(chunks, root=root)
        
        # Convert back to GPU if needed
        if self.backend_manager and self.backend_manager.backend_type == 'gpu':
            local_chunk = self.backend_manager.to_device(local_chunk)
        
        return local_chunk
    
    def gather_array(self, local_array, axis=0, root=0):
        """
        Gather distributed array chunks back to root process.
        
        Parameters
        ----------
        local_array : array-like
            Local array chunk
        axis : int, optional
            Axis along which arrays were split
        root : int, optional
            Rank of root process
        
        Returns
        -------
        array-like or None
            Full array at root, None at other processes
        """
        if not self.is_initialized:
            return local_array
        
        # Convert to CPU if needed
        if self.backend_manager and self.backend_manager.backend_type == 'gpu':
            local_array_cpu = self.backend_manager.asnumpy(local_array)
        else:
            local_array_cpu = local_array
        
        all_chunks = self.gather(local_array_cpu, root=root)

        if self.rank == root:
            full_array = np.concatenate(all_chunks, axis=axis)
            
            # Convert back to GPU if needed
            if self.backend_manager and self.backend_manager.backend_type == 'gpu':
                full_array = self.backend_manager.to_device(full_array)
            
            return full_array
        
        return None
    
    def parallel_compute(self, func, data_list, gather_results=True):
        """
        Perform parallel computation across MPI processes.
        
        Parameters
        ----------
        func : callable
            Function to apply to data
        data_list : list
            List of data items to process (only at rank 0)
        gather_results : bool, optional
            If True, gather all results to rank 0
        
        Returns
        -------
        list or None
            Results at rank 0 if gather_results=True, otherwise local results
        """
        if not self.is_initialized:
            return [func(d) for d in data_list]
        
        if self.rank == 0:
            n_items = len(data_list)
        else:
            n_items = None

        n_items = self.broadcast(n_items, root=0)
        start_idx, end_idx = self.distribute_workload(n_items)

        if self.rank == 0:
            chunks = []
            for i in range(self.size):
                s, e = self._workload_bounds(n_items, rank=i)
                if i == self.rank:
                    continue
                chunk_indices = list(range(s, e))
                chunk_data = [data_list[idx] for idx in chunk_indices]
                chunks.append(chunk_data)
            local_data = [data_list[idx] for idx in range(start_idx, end_idx)]
        else:
            local_data = None
        
        if self.rank != 0:
            local_data = self.recv(source=0, tag=self.rank)
        else:
            for i in range(1, self.size):
                self.send(chunks[i-1], dest=i, tag=i)
        
        local_results = [func(d) for d in local_data]

        if gather_results:
            all_results = self.gather(local_results, root=0)
            if self.rank == 0:
                # Flatten results
                return [item for sublist in all_results for item in sublist]
            return None
        
        return local_results


def setup_mpi_environment(backend_type='auto', device_id=None):
    """
    Set up MPI environment with appropriate backend.
    
    This function initializes MPI and sets up the computational backend,
    automatically assigning GPUs to MPI ranks if available.
    
    Parameters
    ----------
    backend_type : str, optional
        Backend type ('auto', 'cpu', 'gpu')
    device_id : int, optional
        Specific GPU device ID. If None, automatically assigned based on MPI rank.
    
    Returns
    -------
    backend_manager : BackendManager
        Initialized backend manager
    mpi_manager : MPIManager
        Initialized MPI manager
    """
    from .backend_manager import BackendManager
    
    # Initialize MPI first
    mpi_manager = MPIManager()
    
    # Set up backend with MPI awareness
    if device_id is None and backend_type == 'gpu':
        # Automatically assign GPU based on MPI rank
        if mpi_manager.is_initialized:
            try:
                import cupy as cp
                n_gpus = cp.cuda.runtime.getDeviceCount()
                device_id = mpi_manager.rank % n_gpus
            except:
                pass
    
    backend_manager = BackendManager(
        backend_type=backend_type,
        device_id=device_id,
        enable_mpi=mpi_manager.is_initialized,
        multi_gpu=False
    )
    
    mpi_manager.backend_manager = backend_manager
    
    return backend_manager, mpi_manager
