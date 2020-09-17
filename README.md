Geography detection using geography of neighbors

# Algorithm

Cities are detected by multilabel classification (binary cross entropy with logits), assigning most probable cities to a user.
A graph of a social network is taken, and friendship links are used as connections. The graph is made undirected by adding missing links in both directions.
We take users who have cities in their profile, on the basis of which the labeling is built. Number of classes equals to number of cities plus corresponding countries and regions, so that we can detect most probable city along with most probable regeion or a country. Sometimes it/s possible to detect only a region/counry so we take that in worst case. We select cities/regions/countries by choosing a probability threshold of each, say 0.95.
On the basis of the graph, a feature matrix is built: 
a line represents a specific user, a column represents a geo place (country, region or city), in the cell - the counter of meeting of a particular geograhy among his friends.

    | | Russia | Moscow region | Moscow | Sverdlovsk Region | Yekaterinburg | Alapaevsk |	
    | User 1 | 6 | 4 | 4 | 2 | 2 |   |
    | User 2 | 7 | 3 | 3 | 4 | 2 | 2 |

The feature matrix is fed into the GraphSage algorithm, which takes 20 (for example) random friends for each user and takes 5 (for example) random sub friends from each friend. Further, the features of friends and friends of friends are aggregated for this user and added to its initial attributes. A neural network layer acts as an aggregator. Apart from node features edge features were used - how many times one user liked/messaged/mentioned another and used simultaneously with node features in a neural net. For each user, the algorithm simultaneously predicts the probabilities of countries, cities and regions.
	
	| | Russia | Moscow region | Moscow | Sverdlovsk Region | Yekaterinburg | Alapaevsk |
	| User 1 | p01 | p02 | p03 | p04 | p05 | p06 |
	| User 2 | p11 | p12 | p13 | p14 | p15 | p16 |

In addition, in the future, it is planned to add arbitrary features, which may be absent for most users, but counters are always present and work pretty good.
The resulting quality is tens of percent higher compared to heuristic "most frequent city" among friends.

# Data preparation (Rust)

All data is queried from Cassandra, the friends field. LMDB is used to store data: key - user ID, value - list of his friends (binary int32 array). Before saving to the local storage, the graph is made undirected. Graph size is about 250 million users with 30 node degree on average. Apart of VKontakte we maintain Facebook and Twitter graphs.

# Service description

The service consists of three parts:

## lmdb service (Rust)

The service provides a REST API to LMDB and is needed to quickly update a graph, it is used like a graph database. Read (get), add (put) and delete (delete) are supported. Moreover, the put and delete operations retain undirected property of a graph.

Fields:
* db_name - social network name
* keys - list of user IDs
* values - a list of lists of friend IDs that we want to add / remove. Each position in the top-level list corresponds to a position in keys.

Examples:
<pre><code class="bash">

curl -sd '{
	"db_name":"vk",
   	"keys": [11]
}' -H "Content-Type: application/json" -X POST http://localhost:8088/get --fail --silent --show-error | od -w32 -An -tu4 | head

curl -sd '{
	"db_name":"vk",
    "keys": [11],
    "values": [[1,2,9]]
}' -H "Content-Type: application/json" -X POST http://localhost:8088/put

curl -sd '{
	"db_name":"vk",
    "keys": [11],
    "values": [[1,9]]
}' -H "Content-Type: application/json" -X POST http://localhost:8088/delete
</code></pre>

## consumer service (Rust)

The service serves to keep the database up to date and creates a non-persistent geo_detection queue with a limit of 100k in RabbitMQ and processes the following events:

The event_follow and event_unfollow events are passed to the LMDB service and correspond to `put` and `delete` requests.

## Prediction service (Rust, python)

The list of users for whom you need to determine the geography is divided into 10 (for example) big chunks and fed to the neural network in python. For each chunk, a subgraph from the social network is built, namely, for each user random 20 (for example) friends and 5 (for example) random friends of friends are taken. The procedure is the same as described above for GraphSage. Division into subgraphs and prefetching of random friends is done so that each subgraph fits into RAM. If necessary, this allows us to make the algorithm distributed by distributing subgraphs across nodes. It should be noted that LMDB showed excellent performance in parallel construction (reading) of subgraphs and is ideal for random access to a graph, even if it does not fit into RAM. The result of the work is a list of probabilities of belonging to geo locations for each user.

# Tools

While feading features to neural network one should care of the obility to random slicing `sparce` feature matrix. So a fast multithreading (POSIX threads) slicing of a sparce matrix was implemented (see `cymatrix.pyx`) which is several times faster then `scipy.sparse.csr_matrix` slicing.

Label Propagation algorithm was chosen as a baseline model to compare with, parallel implementation of which is in `src/bin/lprop.rs`.

To sort large files on disk external sorter was implemented `src/extsorter.rs`
