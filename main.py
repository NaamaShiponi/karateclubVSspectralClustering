import networkx as nx
import numpy as np
from karateclub import LabelPropagation, SCD, EdMot, GEMSEC
from sklearn.cluster import SpectralClustering
from createDataset import createDataset
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os
import pickle
import glob
import argparse

def load_dataset(graph_file, num_nodes, num_classes, q):
    
    # data = createDataset(num_nodes, num_classes, q)
    # return data.generate_grap()
    if graph_file:
        if os.path.exists(graph_file):
            with open(graph_file, 'rb') as f:
                G, classes = pickle.load(f)
            # print(f"Graph loaded with {len(G.nodes)} nodes and {len(G.edges)} edges,classes {classes}")   
            return G, classes
        else:
            raise FileNotFoundError(f"No saved graph found for num_nodes={num_nodes}, num_classes={num_classes}, q={q}")
    else:
        dataset = createDataset(num_nodes, num_classes, q)
        return dataset.generate_grap(), dataset.classes

def check_karatclub(graph=None):
    model = LabelPropagation()
    model.fit(graph)
    cluster_membership = model.get_memberships()
    return [v for k, v in sorted(cluster_membership.items())]

def check_spectral_clustering(graph, n_clusters):
    adj = nx.adjacency_matrix(graph).toarray()
    sc = SpectralClustering(n_clusters, affinity='precomputed')
    sc.fit(adj)
    return sc.labels_

def accuracy(original_results, try_results):
    original_labels = np.unique(original_results)
    try_labels = np.unique(try_results)
    cost_matrix = np.zeros((len(original_labels), len(try_labels)))
    
    for i, orig_label in enumerate(original_labels):
        for j, try_label in enumerate(try_labels):
            matches = np.sum((original_results == orig_label) & (try_results == try_label))
            cost_matrix[i, j] = -matches
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    label_mapping = {try_labels[j]: original_labels[i] for i, j in zip(row_ind, col_ind)}
    remapped_results = np.array([label_mapping[label] for label in try_results])
    accuracy = np.mean(remapped_results == original_results)
    print(accuracy)
    return accuracy

def run(file,num_nodes, n_classes ,q):
    try:
        graph, classes = load_dataset(file,num_nodes, n_classes,q)
    except FileNotFoundError as e:
        print(e)
        return None, None
    k = check_karatclub(graph)
    sc = check_spectral_clustering(graph, n_classes)
    accuracy_of_karatclub = accuracy(classes, k)
    accuracy_of_spectral_clustering = accuracy(classes, sc)
    return accuracy_of_karatclub, accuracy_of_spectral_clustering

if __name__ == '__main__':
    '''
    Run use DGL graphs:
        python3 main.py --p 0.9 --q 0.3 --grap_dgl_path "/home/naama/.dgl/sbmmixture"
    
    Run without DGL graphs:
        python3 main.py --num_nodes 100 --num_classes 2
        
    Options parameters:
        --num_nodes 100 
        --num_classes 2
    
    '''
    parser = argparse.ArgumentParser(description='GCN Community Detection Parameters')
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--p', type=float, default=0.9, help='Probability p')
    parser.add_argument('--q', type=float, default=0.3, help='Probability q')
    parser.add_argument('--grap_dgl_path', type=str, default=None, help='DGL graphs path')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_classes = args.num_classes
    p = args.p
    q = args.q
    grap_dgl_path= args.grap_dgl_path
   
    if grap_dgl_path: 
        print("DGL graph file")
        folder_path = f'{grap_dgl_path}/graph_{num_nodes}_{num_classes}'
        ave_karatclub_accuracy = 0
        ave_spectral_clustering_accuracy = 0

        # Use glob to get all files in the folder
        files = glob.glob(os.path.join(folder_path, '*'))

        # Loop through each file
        for file in files:
            karatclub_accuracy, spectral_clustering_accuracy = run(file,num_nodes, num_classes, q)
            if karatclub_accuracy is not None and spectral_clustering_accuracy is not None:
                ave_karatclub_accuracy+=karatclub_accuracy
                ave_spectral_clustering_accuracy+=spectral_clustering_accuracy
                
        print(f'Average accuracy of Karatclub: {ave_karatclub_accuracy/len(files)}')
        print(f'Average accuracy of Spectral Clustering: {ave_spectral_clustering_accuracy/len(files)}')   
    else:
        print("New graph")
        maxQ = 0.55
        minQ = 0.05
        step = 0.05
        file =None

        q_values = np.arange(minQ, maxQ, step)
        karatclub_accuracies = []
        spectral_clustering_accuracies = []
        for q in q_values:        
            print('\n---------------------------------')
            print(f'Running this q={q}')
            karatclub_accuracy, spectral_clustering_accuracy = run(file,num_nodes, num_classes, q)
            karatclub_accuracies.append(karatclub_accuracy)
            spectral_clustering_accuracies.append(spectral_clustering_accuracy)
            
        plt.plot(q_values, karatclub_accuracies, label='Label Propagation')
        plt.plot(q_values, spectral_clustering_accuracies, label='Spectral Clustering')
        plt.xlabel('q')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()
    
