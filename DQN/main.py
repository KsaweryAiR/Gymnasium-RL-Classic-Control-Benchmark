import argparse
from config import load_hyperparameters
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description='Train or test a DQN agent.')
    parser.add_argument('hyperparameters', help='Name of the hyperparameter set from hyperparameters.yml')
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--render', action='store_true', help='Render the environment during testing')
    args = parser.parse_args()

    hp = load_hyperparameters('hyperparameters.yml', args.hyperparameters)
    trainer = Trainer(hyperparameter_set=args.hyperparameters, hyperparameters=hp)

    if args.train:
        trainer.train()
    else:
        trainer.test(render=args.render)


if __name__ == '__main__':
    main()
