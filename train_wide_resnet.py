import os
import argparse

from wide_resnet_encoder import WideResNet_Encoder
from wide_resnet import WideResNet


if __name__ == '__main__':

    models = {
        'wide_resnet': WideResNet,
        'encoder': WideResNet_Encoder
    }
    parser = argparse.ArgumentParser(description='Train models on Cifar10')

    parser.add_argument('--model', choices=models.keys(), required=True, help='Specify a model by name to train.')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int)

    args = parser.parse_args()

    model_name = args.model
    args = {k:v for k,v in vars(args).items() if v != None}
    del args['model']

    model = models[model_name](**args, load_weights=False)

    model.train()