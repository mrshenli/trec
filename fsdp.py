import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn

from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.distributed.model_parallel import DistributedModelParallel as DMP
from torchrec.distributed.model_parallel import DataParallelWrapper
from torchrec.distributed.planner.planners import EmbeddingShardingPlanner
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.types import ShardingEnv
from torchrec.modules.embedding_configs import EmbeddingBagConfig


import os
from typing import cast


WORLD_SIZE = 2


def run(rank, world_size):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    num_features = 4
    num_weighted_features = 2
    num_float_features = 10
    batch_size = 3
    device = torch.device(rank)

    ######## create local model ########

    tables = [
        EmbeddingBagConfig(
            num_embeddings=(i + 1) * 10,
            embedding_dim=(i + 1) * 4,
            name="table_" + str(i),
            feature_names=["feature_" + str(i)],
        )
        for i in range(num_features)
    ]

    weighted_tables = [
        EmbeddingBagConfig(
            num_embeddings=(i + 1) * 10,
            embedding_dim=(i + 1) * 4,
            name="weighted_table_" + str(i),
            feature_names=["weighted_feature_" + str(i)],
        )
        for i in range(num_weighted_features)
    ]

    m = TestSparseNN(
        tables=tables,
        num_float_features=num_float_features,
        weighted_tables=weighted_tables,
        dense_device=device,
    ).to(rank)

    if rank == 0:
        for k, v in m.state_dict().items():
            print("--- ", k, v.data.shape)

    ######## wrap with FSDP ########



    fsdp = FullyShardedDataParallel(
        module=m,
        device_id=rank,
        #ignored_modules=[m.sparse],
    )
    if rank == 0:
        print(fsdp)

    ######## run one iteration ########

    _, local_batch = ModelInput.generate(
        batch_size=batch_size,
        world_size=world_size,
        num_float_features=num_float_features,
        tables=tables,
        weighted_tables=weighted_tables,
    )

    batch = local_batch[0].to(rank)
    fsdp(batch)[1].sum().backward()

    sd = fsdp.state_dict()
    for k, v in sd.items():
        print("=== ", k, v.storage().size())
        print(v)


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"

    mp.spawn(run, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)