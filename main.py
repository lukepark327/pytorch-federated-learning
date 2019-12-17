import numpy as np
import random
from pprint import pprint

# from utils.global_dict import GlobalDict
from network.node import Node
from network.graph import TxGraph
from network.transaction import Transaction, TxTypeEnum, Reference, generate_genesis_tx
from network.byzantine import Byzantine, ByzantineType, corrupt_ten_labeled_data
from policy.selection import Selection, SelectionTypeEnum
from policy.updating import Updating, UpdatingTypeEnum
from policy.comparison import Comparison, Metric
from ml.task import Task, create_simple_sequential_model
from ml.flmodel import FLModel
from data.load import load_data
from utils.arguments import parser
from utils.global_dict import GlobalTime

if __name__ == "__main__":
    # Global references
    global_model_dict = dict()
    global_task_dict = dict()
    global_time = GlobalTime()

    # Arguments
    args = parser()
    number_of_nodes = args.nodes
    number_of_rounds = args.rounds
    number_of_epochs = args.epochs
    data_id = args.dataid
    dist_str = args.dist
    eval_rate = args.evalrate
    update_rate = args.updaterate

    # Task definition
    simple_model = create_simple_sequential_model(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        epochs=number_of_epochs
    )
    simple_task = Task(
        'simple_task',
        create_simple_sequential_model,
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        epochs=number_of_epochs,
        model=simple_model
    )
    global_task_dict[simple_task.task_id] = simple_task

    # Data distribution
    x_train, y_train, x_test, y_test = load_data(data_id)
    global_x_test, global_y_test = x_test, y_test
    if dist_str == 'uniform':
        from data.split import split_data_uniform
        x_train = split_data_uniform(x_train, number_of_nodes)
        y_train = split_data_uniform(y_train, number_of_nodes)
        x_test = split_data_uniform(x_test, number_of_nodes)
        y_test = split_data_uniform(y_test, number_of_nodes)
    
    else:
        from data.split import random_choices, split_data_with_choices
        choice_train = random_choices(len(x_train), number_of_nodes)
        choice_test = random_choices(len(x_test), number_of_nodes)
        x_train = split_data_with_choices(x_train, number_of_nodes)
        y_train = split_data_with_choices(y_train, number_of_nodes)
        x_test = split_data_with_choices(x_test, number_of_nodes)
        y_test = split_data_with_choices(y_test, number_of_nodes)

    # Node setting
    # TODO: Make selection, updating, comparison policies organized by config file
    nodes = list()
    genesis_tx = generate_genesis_tx()
    byzantines = np.random.choice(number_of_nodes, 0, replace=False)
    for i in range(number_of_nodes):
        new_txgraph = TxGraph(
            genesis_tx=genesis_tx, 
            eval_set=(x_train[i], y_train[i])
        )
        selection = Selection(
            selection_type=SelectionTypeEnum.HIGH_EVAL_ACC_MODEL,
            owner=str(i),
            number_of_selection=1,
        )
        updating = Updating(
            updating_type=UpdatingTypeEnum.CONTINUAL,
            x_train=x_train[i],
            y_train=y_train[i],
            owner=str(i),
        )
        comparison = Comparison(
            metric=Metric.ACC,
            threshold=0.05,
        )
        new_node = Node(
            nid=str(i),
            global_time=global_time,
            task_id=simple_task.task_id,
            global_model_table=global_model_dict,
            train_set=(x_train[i], y_train[i]),
            test_set=(x_test[i], y_test[i]),
            eval_rate=eval_rate,
            tx_graph=new_txgraph,
            selection=selection,
            updating=updating,
            comparison=comparison
        )
        if i in byzantines:
            byzantine = Byzantine(
                byzantine_type=ByzantineType.CORRUPTED_DATA_SET
            )
            new_node.byzantine = byzantine
            corrupt_ten_labeled_data(x_train[i], y_train[i], 4, 9)

        # Each node has its own local trained result
        # The results are recorded on global_model_dict['local-(node_id)']
        nodes.append(new_node)

    # Set adjacent nodes
    # TODO: Random 
    num_of_adjacent = 5

    for i in range(number_of_nodes):
        choices = np.random.choice(number_of_nodes, num_of_adjacent, replace=False)
        nodes[i].adjacent_list = [ nodes[j] for j in choices if j.item is not i ]
    
    # Tick!
    global_time.tick()

    # Make the first task transaction by node 0
    open_tx = nodes[0].open_task(task=simple_task)
    print(nodes[0])

    # Simulate the network
    for i in range(number_of_rounds):
        print("### Round: ", i)
        for node in nodes:
            node.send_txs_in_buffer()
        
        for node in nodes:
            # If node has not made any model, train locally
            if node.model_id is None:
                if node.tx_graph.has_transaction(open_tx):
                    node.init_local_train(
                        task=simple_task,
                        open_tx=open_tx,
                        tx_making_rate=update_rate
                    )
            # Node updates its model with probability
            if random.random() < update_rate:
                node.update(simple_task)

        for node in nodes:
            node.get_transactions_from_buffer()
        
        global_time.tick()
    
    model_set = set()
    eval_dict = dict()
    
    for node in nodes:
        if node.current_model is not None:
            print(node)
            print(global_model_dict[node.model_id].evaluate(global_x_test, global_y_test))
            model_set.add(node.model_id)

    print("model set")
    print(len(model_set))
