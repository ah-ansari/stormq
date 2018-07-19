import numpy as np
import networkx as nx
import tools
import feature
import train
import competition
import run

run.id = 0
run.data_set = "CA-GrQc.txt"
run.data_set_directed = False
run.train_n_rounds = 700
run.opponent = 0

run.description = "complete base paper" \
                  "for t=21 means one round in each train round" \
                  "feature getting 1000 times" \
                  "actions: degree-weight-blocking\n" \
                  "states: original paper states\n" \
                  "epsilon being tuned\n" \
                  "selecting random in unseen states in competition\n" \
                  "700 round of training\n" \
                  "delayed reward: t%6 == 0\n" \
                  "alpha = 0.5 - eps = 0.8 - gamma = 0.98 - d = 0.998   \n" \
                  "eps is tuned: eps = u - ((u-l)*t/max,alpha tuned: alpha=alpha*d\n"

run.save_initial()

if run.data_set_directed:
    G = tools.load_graph_directed("datasets\\"+run.data_set)
else:
    G = tools.load_graph_undirected("datasets\\"+run.data_set)
nx.write_gpickle(G,"graph")

# features
G = nx.read_gpickle("graph")
features = feature.get_features_values_by_sampling(G)
ranges = feature.get_ranges(features)
np.save(str(run.id)+"/ranges", ranges)

for i in range(run.n_run):
    G = nx.read_gpickle("graph")
    print("run: "+str(i)+"-------------------------")
    q_table = train.train(G, ranges)

    # competition
    result = competition.compete(G, q_table, ranges)
    run.save_one_run(i, q_table, result)
