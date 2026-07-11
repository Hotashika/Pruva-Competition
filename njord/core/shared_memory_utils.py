import time
from multiprocessing import resource_tracker, shared_memory


def attach_existing_shared_memory(name, retries=50, delay=0.1, track=False):
    last_error = None
    for _ in range(retries):
        try:
            shm = shared_memory.SharedMemory(name=name)
            if not track:
                _unregister_from_resource_tracker(shm)
            return shm
        except FileNotFoundError as exc:
            last_error = exc
            time.sleep(delay)

    raise RuntimeError(f"{name} shared memory not found") from last_error


def _unregister_from_resource_tracker(shm):
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass


def close_shared_memory_handles(*handles):
    for shm in handles:
        if shm is not None:
            shm.close()
