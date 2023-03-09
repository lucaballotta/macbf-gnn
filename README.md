# MACBF-GNN

Learning decentralized control barrier functions using graph neural networks.

## Dependencies

To install the requirements:

```bash
conda create -n macbf-gnn python=3.9
conda activate macbf-gnn
pip install -r requirements.txt
```

Then you need to install the torch_geometric package following the [official website](https://pytorch-geometric.readthedocs.io/en/latest/).

## Train

To train the model, use:

```bash
python train.py --env SimpleCar -n 10 --steps 500000
```

One can refer to [`settings.yaml`](settings.yaml) for the training parameters. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`

## Test

To test the learned model, use:

```bash
python test.py --path <path-to-log> --epi <number-of-episodes>
```

For large-scale tests, one can also use:

```bash
bash test.sh <path-to-log> <number-of-episodes>
```

One can add `1` to the arguments if one wants to generate videos. After the large-scale test, one can use the following command to calculate the safe rate:

```bash
python safe_rate.py --path <path-to-log>
```

