import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchrec.distributed.model_parallel import DistributedModelParallel
from torch.distributed.fsdp import FullyShardedDataParallel

from torchrec.distributed.test_utils.test_model import ModelInput, TestSparseNN
from torchrec.modules.embedding_configs import EmbeddingBagConfig


import os


def run(rank, world_size):

    print("0000", rank, world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    num_features = 4
    num_weighted_features = 2
    num_float_features = 10
    batch_size = 3

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

    print("1111")
    m = TestSparseNN(
        tables=tables,
        num_float_features=num_float_features,
        weighted_tables=weighted_tables,
        dense_device=torch.device(rank),
    ).to(rank)

    print("2222")
    _, local_batch = ModelInput.generate(
        batch_size=batch_size,
        world_size=1,
        num_float_features=num_float_features,
        tables=tables,
        weighted_tables=weighted_tables,
    )
    batch = local_batch[0].to(rank)
    print("3333")
    m(batch)[1].sum().backward()
    print("4444")


if __name__=="__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29506"

    world_size = 1
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)