from pathlib import Path
import numpy as np
import torch
import argparser
from loss import loss_fn
from model import DisentangledVAE
from trainer import DisentangleSeqVaeTrainer
from dataset import SpriteDataset
from dataloader import MyDataLoader

NUM_WORKER = 0
SHUFFLE = True  # this is turned false if args.valid_split != 0


def make_determine():
    np.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    make_determine()
    # if args.resume, arguments are ignored and overriden from what they originally were.
    if args.resume is not None:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            raise FileNotFoundError("The specified resume_path is not found: %s" % str(resume_path))
        args = torch.load(str(resume_path))['args']

    sprite_dataset = SpriteDataset(args.path_data, args.debug)
    valid_idx = list(range(9000, len(sprite_dataset)))
    tr_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': NUM_WORKER,
                 'validation_split': 0.0, 'validation_idx': valid_idx}

    # initialize a same model for all folds (if CV = True)
    disenVAE = DisentangledVAE(f_dim=args.f_dim, z_dim=args.z_dim, step=args.n_channel, factorised=args.factorized)
    print(disenVAE)
    optimizer = torch.optim.Adam(disenVAE.parameters(), lr=0.0002)

    train_loader = MyDataLoader(sprite_dataset, **tr_params)
    valid_loader = train_loader.split_validation()

    trainer = DisentangleSeqVaeTrainer(model=disenVAE, loss=loss_fn, optimizer=optimizer, args=args,
                                       data_loader=train_loader, valid_data_loader=valid_loader)
    trainer.train()


if __name__ == '__main__':
    my_arg_parser = argparser.ArgParser()
    args = my_arg_parser.parse()
    print('Add arguments ...')
    print(args)
    main(args)
