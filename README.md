# ANGraph: A GNN-Based Performance Prediction Framework for Asynchronous Neuromorphic Hardware

## About project
In this project, we propose a novel performance prediction framework for asynchronous neuromorphic hardware, which transforms the intermediate representation (IR) of the TrueAsync simulator into asynchronous neuromorphic graphs and predicts hardware performance including latency and power consumption.

[fig1.pdf](https://github.com/user-attachments/files/18514453/fig1.pdf)


### Contribution
* Transforming: we introduce a method to analyze event streams and define asynchronous neuromorphic graphs as IRs. By transforming the event streams into graphs, we bridge system-level simulation and graph-level representations, enabling GNN training and inference for hardware latency predictions.
* Benchmarking: we design register-transfer level (RTL) and netlist circuits for asynchronous neuromorphic hardware at various scales and manufacturing process nodes. These gate-level netlists are simulated using diverse compiled input data from different applications. By pairing system-level event streams from the TrueAsync simulator with gate-level hardware performance results, we build a standard benchmark suite with more than one million samples, including hardware latency and power consumption prediction tasks.
* Predicting: we put forward the GNN-based ANGraph-L and the residual network (ResNet)-based ANGraph-P models for predicting hardware latency and power consumption, respectively. Compared to the latency prediction results from the TrueAsync simulator, we improve the $R^2$ score by 0.69 and reduce RMSE by 78\%. Furthermore, we evaluate these two models on additional datasets without extra training across different scales, input traffic patterns, and process nodes. ANGraph-L achieves an average $R^2$ score of 0.96, while ANGraph-P achieves an average $R^2$ score of 0.98 and a MAPE of 0.88\%. 

##

<!-- Benchmarking -->
## Dataset
All of our datasets are stored in the [data](./data) folder.
### Latency Prediction Dataset
We have constructed thirteen latency prediction task datasets, corresponding to different scales, input traffic patterns, and process nodes. Our datasets are modeled as graphs, and each dataset includes the following files:
* edge.csv.gz: Describes the connection relationships between different nodes.
* edge-feat.csv.gz: The feature vector of each edge.
* graph-label.csv.gz: The label value of each graph, which is the latency of the circuit.
* node-feat.csv.gz: The feature vector of each node.
* num-edge-list.csv.gz: The number of edges.
* num-node-list.csv.gz: The number of nodes.

The aforementioned files are stored in the raw folder of each dataset and can be viewed at your convenience.


### Power Prediction Dataset  
We have constructed six power prediction datasets. These datasets are relatively simple, with each dataset mainly comprising the following two files:
* feature.npy: The feature vector of the circuit's power consumption.
* label.npy: The three energy consumption values of the circuit, corresponding to dynamic energy consumption, static energy consumption, and total energy consumption respectively.

<!-- Modeling and Advancing -->
## Modeling
 We propose a GNN-based ANGraph-L model for latency predictions to learn from asynchronous neuromorphic graphs. In addition, since the power consumption prediction task deals with activity matrices instead of graphs, we propose a ResNet-based ANGraph-P for power consumption predictions.

[fig4.pdf](https://github.com/user-attachments/files/18514454/fig4.pdf)


### ANGraph-L 
* You can run [ANGL_train.py](ANGL_train.py) to train ANGraph-L on the specified dataset.
* You can run [ANGL_test.ipynb](ANGL_test.ipynb) to evaluate the model's performance across different scales and input traffic patterns.
* You can run [ANGL_transfer.py](ANGL_transfer.py) to perform transfer learning and obtain models for different process nodes.
### ANGraph-P
* You can run [ANGP_train.ipynb](ANGP_train.ipynb) to train and test ANGraph-P.
* The trained models are stored in the [saved_model](saved_model) folder.
* The training process files are saved in folder [logs](./logs).
