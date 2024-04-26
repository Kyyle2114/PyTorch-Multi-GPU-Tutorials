"""
Pytorch DDP Code Reference : https://mvje.tistory.com/141#recentComments
"""

import argparse
import numpy as np
import time

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from data import dataset
from model import resnet50
from util import seed, trainer

from torchinfo import summary

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--port', type=int, default=2024)
    parser.add_argument('--local_rank', type=int)
    return parser

def init_distributed_training(rank, opts):
    # 1. setting for distributed training
    opts.rank = rank
    opts.gpu = opts.rank % torch.cuda.device_count()
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    torch.distributed.init_process_group(backend='nccl',
                            init_method='tcp://127.0.0.1:' + str(opts.port),
                            world_size=opts.ngpus_per_node,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()

    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print('opts :', opts)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main(rank, opts):
    seed.seed_everything(21)  
    
    init_distributed_training(rank, opts)
    local_gpu_id = opts.gpu

    train_set, val_set = dataset.load_CIFAR10()

    ### Train / Validation set ###    
    train_sampler = DistributedSampler(dataset=train_set, shuffle=True)
    batch_sampler_train = torch.utils.data.BatchSampler(train_sampler, opts.batch_size, drop_last=True)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler_train, num_workers=opts.num_workers)
    
    val_loader = DataLoader(val_set, batch_size=opts.batch_size, shuffle=False)

    ### Model ###
    model = resnet50.ResNet50().cuda(local_gpu_id)
    
    print()
    print('=== MODEL INFO ===')
    summary(model)
    print()

    model = DistributedDataParallel(module=model, device_ids=[local_gpu_id])
    
    ### Training config ### 
    criterion = torch.nn.CrossEntropyLoss().to(local_gpu_id)
    optimizer = torch.optim.Adam(model.parameters())

    EPOCH = 10
    max_loss = np.inf

    for epoch in range(EPOCH):

        train_sampler.set_epoch(epoch)

        train_loss, train_acc = trainer.model_train(
            model=model, 
            data_loader=train_loader, 
            criterion=criterion, 
            optimizer=optimizer, 
            device=local_gpu_id
        )
                
        val_loss, val_acc = trainer.model_evaluate(
            model=model.module, 
            data_loader=val_loader, 
            criterion=criterion, 
            device=torch.device(f"cuda:{local_gpu_id}")
        )
        
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)            
        train_loss = train_loss.item() / dist.get_world_size()
        
        dist.all_reduce(train_acc, op=dist.ReduceOp.SUM)
        train_acc = train_acc.item() / dist.get_world_size()

        if (val_loss < max_loss) and (opts.rank == 0):
            print(f'[INFO] val_loss has been improved from {max_loss:.5f} to {val_loss:.5f}. Save model.')
            max_loss = val_loss
            torch.save(model.state_dict(), 'Best_Model_DDP.pth')

        print(f'epoch {epoch+1:02d}, loss: {train_loss:.5f}, accuracy: {train_acc:.5f}, val_loss: {val_loss:.5f}, val_accuracy: {val_acc:.5f} \n')

    print('=== DONE === \n')    

if __name__ == '__main__':
    
    DOWNLOAD_CIFAR10 = True
    
    if DOWNLOAD_CIFAR10:
        dataset.download_CIFAR10()

    parser = argparse.ArgumentParser('Distributed training test', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    # ngpus_per_node = 2
    opts.ngpus_per_node = torch.cuda.device_count()
    
    # gpu_ids = 0, 1
    opts.gpu_ids = list(range(opts.ngpus_per_node))
    opts.num_workers = opts.ngpus_per_node * 4

    start_time = time.time()
    
    torch.multiprocessing.spawn(
        main,
        args=(opts,),
        nprocs=opts.ngpus_per_node,
        join=True)
    
    end_time = time.time()
    
    print('Elapsed time %s'%(end_time - start_time))