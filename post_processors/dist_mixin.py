import torch.distributed as dist
from typing import List, Any


class DistGatherMixin:
    def gather(self):
        pass

    @staticmethod
    def gather_object(objects: List[Any]):
        output = [None for _ in range(dist.get_world_size())]
        dist.gather_object(objects,
                           object_gather_list=output if dist.get_rank() == 0 else None,
                           dst=0)

        if dist.get_rank() == 0:
            return output
        else:
            return None
