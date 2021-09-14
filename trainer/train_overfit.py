"""trainer utility for 3DCNN"""

from pathlib import Path

import torch

from dataset.mesh_data import MeshData
from model.net_texture import NetTexture
from model.siren.modules import SingleBVPNet


def main(config):
    """
    Driver function for training
    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
    """

    # declare device
    device = torch.device('cpu')
    if torch.cuda.is_available() and config['device'].startswith('cuda'):
        device = torch.device(config['device'])
        print('Using device:', config['device'])
    else:
        print('Using CPU')

    # create dataloaders
    trainset = MeshData()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    valset = MeshData()

    # instantiate model
    # MLP with Positional encoding
    model = NetTexture(sample_point_dim=3, texture_features=3, n_layers=8, n_freq=10, ngf=256)

    # SIREN
    # model = SingleBVPNet(type='sine', in_features=3, out_features=3)

    # load model if resuming from checkpoint
    if config['resume_ckpt'] is not None:
        model.load_state_dict(torch.load(config['resume_ckpt'], map_location='cpu'))

    # move model to specified device
    model.to(device)

    # create folder for saving checkpoints
    Path(f'runs/{config["experiment_name"]}').mkdir(exist_ok=True, parents=True)
    Path(f'runs/{config["experiment_name"]}/outputs').mkdir(exist_ok=True, parents=True)

    # start training
    train(model, trainloader, valset, device, config)


def train(model, trainloader, valds, device, config):

    # declare loss and move to specified device
    loss_criterion = torch.nn.L1Loss(reduction='mean')
    loss_criterion.to(device)

    # declare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config['max_epochs']):

        for i, batch in enumerate(trainloader):
            # move batch to device
            MeshData.move_batch_to_gpu(batch, device)

            # zero out previously accumulated gradients
            optimizer.zero_grad()

            # forward pass
            prediction = model(batch['vertex'])

            # compute total loss = sum of loss for whole prediction + losses for partial predictions
            loss_total = loss_criterion(prediction, batch['color'])

            # compute gradients on loss_total
            loss_total.backward()

            # update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(trainloader) + i

            if iteration % config['print_every_n'] == (config['print_every_n'] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running / config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation evaluation and logging
            if iteration % config['validate_every_n'] == (config['validate_every_n'] - 1):

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                model.eval()

                # forward pass and evaluation for entire validation set

                vertex, color = valds.get_data_on_device(device)

                with torch.no_grad():
                    prediction = model(vertex)

                loss_total_val = loss_criterion(prediction, color).item()

                valds.visualize(prediction.cpu().numpy(), f"runs/{config['experiment_name']}/outputs", epoch)

                print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_total_val:.3f}')

                torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')

                # set model back to train
                model.train()


if __name__ == '__main__':
    config = {
        'experiment_name': 'overfit_colors',
        'device': 'cuda:0',  # change this to cpu if you do not have a GPU
        'batch_size': 128,
        'resume_ckpt': None,
        'learning_rate': 0.0005,
        'max_epochs': 5000,
        'print_every_n': 100,
        'validate_every_n': 2500,
    }

    main(config)
