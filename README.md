# DDL-simulator
Decentralized Deep Learning Simulator.

Keywords:
* Federated Learning
* Privacy Preserving Deep Learning

# TODO
- [ ] DDL-blockchain
- [ ] Packaging .py files
- [ ] easy installation about dependencies

# How to Use

For example, you can run `main.py` with the following code:

```bash
python main.py --nodes=5 --round=100
```

The `main.py` does simple DDL simulation with several policies:

* `my_policy_update_model_weights` updates DL model's weight with simple averaged weight among all nodes.

* `equally_fully_connected` makes fully-connected network.

* `my_policy_update_txs_weight` defines how to update transaction's weight in DAG. Each transaction updates its all predecessors' weight for adding its weight.

# Basic Simulation Flow

* Get arguments with `arguments.parser()`.

* Load entire dataset.

* Split the dataset into many chunks.
  * We will distribute each chunk to nodes.
  * Leave one for master testset.

* Set nodes.

* Set blockchain.

* Repeat **rounds** sufficiently.
  * Each node trains, tests, updates (its weights), and send transaction(s) per round.

# License

The DDL-simulator project is licensed under the MIT License, also included in our repository in the [LICENSE](https://github.com/lukepark327/DDL-simulator/blob/master/LICENSE) file.
