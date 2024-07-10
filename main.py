import networkx as nx
import numpy as np
from karateclub import LabelPropagation
from sklearn.cluster import SpectralClustering
from createDataset import createDataset
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import os
import pickle
import glob
import argparse
import csv

def load_dataset(graph_file, num_nodes, num_classes, q):
    if graph_file:
        if os.path.exists(graph_file):
            with open(graph_file, 'rb') as f:
                G, classes = pickle.load(f)
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
    return accuracy

def run(file, num_nodes, n_classes, q):
    try:
        graph, classes = load_dataset(file, num_nodes, n_classes, q)
    except FileNotFoundError as e:
        print(e)
        return None, None, None, None, None, None
    k = check_karatclub(graph)
    sc = check_spectral_clustering(graph, n_classes)
    accuracy_of_karatclub = accuracy(classes, k)
    accuracy_of_spectral_clustering = accuracy(classes, sc)
    return graph, classes, accuracy_of_karatclub, k, accuracy_of_spectral_clustering, sc

if __name__ == '__main__':
    '''
    Run use DGL graphs:
        python3 main.py --p 0.6 --q 0.3 --grap_dgl_path "/home/naama/.dgl/sbmmixture" --run_number 000
    
    Run without DGL graphs:
        python3 main.py --run_number 000
        
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
    parser.add_argument('--run_number', type=str, required=True, help='Run number for this execution')

    args = parser.parse_args()

    num_nodes = args.num_nodes
    num_classes = args.num_classes
    p = args.p
    q = args.q
    grap_dgl_path = args.grap_dgl_path
    run_number = args.run_number

    results = []
    
    # create a folder to save the results
    if not os.path.exists('results'):
        os.makedirs('results')
    
    if grap_dgl_path:
        folder_path = f'{grap_dgl_path}/graph_{num_nodes}_{num_classes}'
        ave_karatclub_accuracy = 0
        ave_spectral_clustering_accuracy = 0
        files = glob.glob(os.path.join(folder_path, '*'))

        for file in files:
            graph, classes, karatclub_accuracy, k, spectral_clustering_accuracy, sc = run(file, num_nodes, num_classes, q)
            if karatclub_accuracy is not None and spectral_clustering_accuracy is not None:
                ave_karatclub_accuracy += karatclub_accuracy
                ave_spectral_clustering_accuracy += spectral_clustering_accuracy
                results.append({
                    "grap_dgl_path": file,
                    "p":p,
                    "q":q,
                    "accuracy_of_karatclub": karatclub_accuracy,
                    "karatclub_classes": k,
                    "accuracy_of_spectral_clustering": spectral_clustering_accuracy,
                    "sc_classes": sc,
                    "classes": classes
                })

        ave_karatclub_accuracy /= len(files)
        ave_spectral_clustering_accuracy /= len(files)
        results.append({
            "ave_karatclub_accuracy": ave_karatclub_accuracy,
            "ave_spectral_clustering_accuracy": ave_spectral_clustering_accuracy
        })
        print(f'Average accuracy of Karatclub: {ave_karatclub_accuracy}')
        print(f'Average accuracy of Spectral Clustering: {ave_spectral_clustering_accuracy}')
    else:
        maxQ = 0.55
        minQ = 0.05
        step = 0.05
        file = None

        q_values = np.arange(minQ, maxQ, step)
        karatclub_accuracies = []
        spectral_clustering_accuracies = []
        for q in q_values:
            print('\n---------------------------------')
            print(f'Running this q={q}')
            graph, classes, karatclub_accuracy, k, spectral_clustering_accuracy, sc = run(file, num_nodes, num_classes, q)
            karatclub_accuracies.append(karatclub_accuracy)
            spectral_clustering_accuracies.append(spectral_clustering_accuracy)
            results.append({
                "q": q,
                "p": p,
                "accuracy_of_karatclub": karatclub_accuracy,
                "karatclub_classes": k,
                "accuracy_of_spectral_clustering": spectral_clustering_accuracy,
                "sc_classes": sc,
                "classes": classes
            })

        plt.plot(q_values, karatclub_accuracies, label='Label Propagation')
        plt.plot(q_values, spectral_clustering_accuracies, label='Spectral Clustering')
        plt.xlabel('q')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'results/karatclub_plt_{run_number}.png')
        plt.show()
        

    with open(f'results/karatclub_{run_number}.csv', 'w', newline='') as csvfile:
        fieldnames = [
            "p", "q" ,
            "accuracy_of_karatclub", "accuracy_of_spectral_clustering",
            "karatclub_classes", "sc_classes",
            "ave_karatclub_accuracy", "ave_spectral_clustering_accuracy","classes","grap_dgl_path"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = {
                "grap_dgl_path": result.get("grap_dgl_path"),
                "p": result.get("p"), 
                "q": result.get("q"),
                "classes": result.get("classes"),
                "accuracy_of_karatclub": result.get("accuracy_of_karatclub"),
                "karatclub_classes": result.get("karatclub_classes"),
                "accuracy_of_spectral_clustering": result.get("accuracy_of_spectral_clustering"),
                "sc_classes": result.get("sc_classes"),
                "ave_karatclub_accuracy": result.get("ave_karatclub_accuracy"),
                "ave_spectral_clustering_accuracy": result.get("ave_spectral_clustering_accuracy")
            }
            writer.writerow({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()})
