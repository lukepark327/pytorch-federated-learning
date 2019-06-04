# PPDL
Privacy-Preserving-Deep-Learning based on Federated Learning

## How to use

```bash
python main.py --dist "random" --bias "random" --nodes=100 --round=10000 --dataset=58000
```

or

```bash
sh ./run.sh
```

### help
```bash
python main.py --help
usage: main.py [-h] [--nodes N] [--overlap O] [--dist D] [--bias D]

Process some integers.

optional arguments:
  -h, --help   show this help message and exit
  --nodes N    total number of nodes
  --overlap O  portion of overlapping data (allow no overlap: 1.0)
  --dist D     mechanism of total data distribution
  --bias B     mechanism of one's data distribution
```

## Visualization
![](https://github.com/twodude/PPDL/blob/master/images/dist.png)

## References
TBA
