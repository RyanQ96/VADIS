"""
Circular Self-Organizing Map (SOM) Implementation

This module implements a Circular Self-Organizing Map (SOM) algorithm for projecting high-dimensional data onto a 2D circular layout. 
It extends the traditional SOM by incorporating a relevance score for each data point, positioning high-relevance data near the center of the circle, and balancing the influence of relevance and similarity during training.

Key Features:
- Circular neuron grid layout.
- Balancing of relevance and similarity in training.
- Positioning of high-relevance data points near the center.
- Compatible with high-dimensional data for dimensionality reduction.

References:
- This implementation is based on the Self-Organizing Map (SOM) algorithm from the 
  MiniSom project (minisom.py). The original MiniSom implementation has been extended 
  to support circular layouts and GPU acceleration.
  - Original MiniSom Project: https://github.com/JustGlowing/minisom
- SOM: T. Kohonen, Self-Organizing Maps, Springer, 1995 
"""

import numpy as np
from sklearn.manifold import TSNE
import math
from time import time
from datetime import timedelta
from sys import stdout
import pandas as pd


def _build_iteration_indexes(data_len, num_iterations, verbose=False, random_generator=None, use_epochs=False, client_handler=None, card_id=None):
    """Returns an iterable with the indexes of the samples to pick at each iteration of the training.

    If random_generator is not None, it must be an instance of numpy.random.RandomState and it will be used to randomize the order of the samples.
    """
    if use_epochs:
        iterations = []
        for _ in range(num_iterations):
            iterations_per_epoch = np.arange(data_len)
            if random_generator:
                random_generator.shuffle(iterations_per_epoch)
            iterations.append(iterations_per_epoch)
    else:
        iterations = np.arange(num_iterations) % data_len
        if random_generator:
            random_generator.shuffle(iterations)
    if verbose:
        return _wrap_index_in_verbose(iterations, client_handler=client_handler, card_id=card_id)
    else:
        return iterations


def _wrap_index_in_verbose(iterations, client_handler=None, card_id=None):
    """Yields the values in iterations, printing the status on the stdout."""
    m = len(iterations)
    digits = len(str(m))
    progress = '\r [ {s:{d}} / {m} ] {s:3.0f}% - ? it/s'.format(m=m, d=digits, s=0)
    stdout.write(progress)
    beginning = time()
    for i, it in enumerate(iterations):
        yield it
        sec_left = ((m - i + 1) * (time() - beginning)) / (i + 1)
        time_left = str(timedelta(seconds=sec_left))[:7]
        progress = '\r [ {i:{d}} / {m} ] {p:3.0f}% - {time_left} left '.format(
            i=i + 1, d=digits, m=m, p=100 * (i + 1) / m, time_left=time_left)
        if client_handler:
            client_handler.emitTaskEnvLayoutComputationProgress({"progress": 100 * (i + 1) / m, "cardId": card_id})
        stdout.write(progress)


def asymptotic_decay(learning_rate, t, max_iter):
    """Decay function of the learning process."""
    return learning_rate / (1 + t / (max_iter / 2))


def get_best_number_of_layers(step: int, data_length: int, skip=1):
    """Get the best number of layers for the circular SOM.

    Parameters
    ----------
    step : int
        The number of neurons in the first layer.

    data_length : int
        The number of samples in the dataset.

    skip : int
        The number of layers to skip.

    Returns
    -------
    int
        The best number of layers for the circular SOM.
    """
    num_layers = 1
    num_neurons = step * (num_layers + skip)
    while num_neurons < data_length:
        num_layers += 1
        num_neurons += step * (num_layers + skip)
    return num_layers + 1


def generate_circular_grid(step, layer, radius=1, skip=1):
    """Generates a circular grid for the SOM."""
    pi = math.pi
    points = []
    weights_radius = []
    points_map = {}
    global_index = 0
    num_of_points_per_layer = {}
    radius_per_layer = {}

    for l in range(layer):
        num_in_current_layer = step + step * (l + skip)
        num_of_points_per_layer[l] = num_in_current_layer
        angle_interval = 2 * pi / num_in_current_layer
        radius_in_current_layer = (l + 1 + skip) * radius
        for i in range(num_in_current_layer):
            angle = angle_interval * i - pi / num_in_current_layer
            point = [
                radius_in_current_layer * math.cos(angle),
                radius_in_current_layer * math.sin(angle),
            ]
            points.append(point)
            weights_radius.append(radius_in_current_layer)
            points_map[str(tuple(point))] = {
                "coords": point,
                "layer": l,
                "angle": angle,
                "radius": radius_in_current_layer,
                "index": global_index
            }
            global_index += 1
        radius_per_layer[l] = radius_in_current_layer

    weights_radius = np.exp(1 / np.array(weights_radius))
    weights_radius = (weights_radius - np.min(weights_radius)) / (np.max(weights_radius) - np.min(weights_radius))

    return np.array(points), weights_radius, points_map, num_of_points_per_layer, radius_per_layer


class CircularSOM:
    """Circular Self-Organizing Map (SOM) implementation."""

    def __init__(self, step, layer, input_len, sigma=1.0, learning_rate=0.5, decay_function=asymptotic_decay, neighborhood_function='gaussian', topology='circular', activation_distance='euclidean', random_seed=None):
        """Initializes the Circular SOM.

        Parameters
        ----------
        step : int
            Number of neurons in the first layer.

        layer : int
            Number of layers in the circular grid.

        input_len : int
            Dimensionality of the input data.

        sigma : float
            Initial neighborhood value.

        learning_rate : float
            Initial learning rate.

        decay_function : function
            Function that reduces learning_rate and sigma at each iteration.

        neighborhood_function : str
            Type of neighborhood function ('gaussian' supported).

        topology : str
            Type of topology ('circular' supported).

        activation_distance : str
            Type of distance function ('euclidean' supported).

        random_seed : int
            Random seed for reproducibility.
        """
        self._random_generator = np.random.RandomState(random_seed)
        self._learning_rate = learning_rate
        self._sigma = sigma
        self._input_len = input_len
        self.topology = topology

        # Generate the circular grid
        self._grid, self._weights_radius, self._grid_map, self._num_of_points_per_layer, self._radius_per_layer = generate_circular_grid(step, layer)
        self._grid = self._grid.astype(float)
        self._total_cells = len(self._grid)
        self._weights = self._random_generator.rand(self._total_cells, input_len) * 2 - 1
        self._weights /= np.linalg.norm(self._weights, axis=-1, keepdims=True)

        self._activation_map = np.zeros((self._total_cells,))
        self.initialize_resource_map()
        self._decay_function = decay_function

        # Set neighborhood function
        if neighborhood_function == 'gaussian':
            self.neighborhood = self._gaussian
        else:
            raise ValueError(f"Neighborhood function '{neighborhood_function}' is not supported.")

        # Set activation distance function
        if activation_distance == 'euclidean':
            self._activation_distance = self._euclidean_distance
        else:
            raise ValueError(f"Activation distance '{activation_distance}' is not supported.")

    def initialize_resource_map(self):
        """Initializes the resource map."""
        self._resource_map = np.ones((self._total_cells,))

    def update_weights_radius_with_relevance(self, relevance):
        """Updates the weights radius with the relevance scores."""
        num_processed = 0
        new_weights_relevance = []
        sorted_relevance = np.sort(relevance)[::-1]
        for l in range(len(self._num_of_points_per_layer)):
            num_of_points = self._num_of_points_per_layer.get(l)
            if num_of_points is not None and num_of_points > 0:
                target_values = sorted_relevance[num_processed:num_processed + num_of_points]
                if len(target_values) > 0:
                    mean_relevance = np.mean(target_values)
                    new_weights_relevance += [mean_relevance] * num_of_points
                else:
                    new_weights_relevance += [0] * num_of_points
                num_processed += num_of_points
        self._weights_radius = np.array(new_weights_relevance)

    def train(self, data, relevance_score, num_iteration, w_s=0.4, w_r=0.6, random_order=False, client_handler=None, card_id=None, verbose=False, use_epochs=False, report_error=False, use_sorted=False):
        """Trains the SOM by picking samples at random from data.

        Parameters
        ----------
        data : array-like, shape = [n_samples, n_features]
            The input data.

        relevance_score : array-like, shape = [n_samples]
            Relevance scores corresponding to the data.

        num_iteration : int
            Number of iterations for training.

        w_s : float
            Weight for similarity.

        w_r : float
            Weight for relevance.

        random_order : bool
            Whether to randomize the order of data samples.

        verbose : bool
            Whether to print progress.

        use_epochs : bool
            Whether to use epochs.

        report_error : bool
            Whether to report error during training.

        use_sorted : bool
            Whether to sort data based on relevance.
        """
        self._check_iteration_number(num_iteration)
        self._check_input_len(data)
        self.epoch_cur = 1
        self._wr = w_r
        self._ws = w_s
        random_generator = None
        if random_order:
            random_generator = self._random_generator
        iterations = _build_iteration_indexes(len(data), num_iteration, verbose, random_generator, use_epochs, client_handler=client_handler, card_id=card_id)
        if use_epochs:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index / data_len)
        else:
            def get_decay_rate(iteration_index, data_len):
                return int(iteration_index)
        self.update_weights_radius_with_relevance(relevance_score)
        if use_sorted:
            sorted_data = sorted(zip(data, relevance_score), key=lambda t: t[1], reverse=True)
            data = [x for x, r in sorted_data]
            relevance_score = [r for x, r in sorted_data]
        if use_epochs:
            for epoch_num, iteration_idx in enumerate(iterations):
                for t, iteration in enumerate(iteration_idx):
                    decay_rate = get_decay_rate(t, len(data))
                    winner = self.winner(data[iteration], relevance_score[iteration])
                    self.update(data[iteration], relevance_score[iteration], winner, decay_rate, num_iteration)
                if report_error:
                    self.dual_quantization_error(data, relevance_score)
                    self.start_new_epoch(verbose)
        else:
            for t, iteration in enumerate(iterations):
                decay_rate = get_decay_rate(t, len(data))
                if iteration == 0:
                    if report_error:
                        self.dual_quantization_error(data, relevance_score)
                    self.start_new_epoch(verbose)
                winner = self.winner(data[iteration], relevance_score[iteration])
                self.update(data[iteration], relevance_score[iteration], winner, decay_rate, num_iteration)

    def dual_quantization_error(self, data, relevance, sort=True):
        """Computes the dual quantization error."""
        self.initialize_resource_map()
        total_loss = 0
        if sort:
            modified_data = sorted(zip(data, relevance), key=lambda t: t[1], reverse=True)
        else:
            modified_data = zip(data, relevance)
        for x, r in modified_data:
            _, loss = self.winner(x, r, return_distance=True)
            total_loss += loss
        print(f"\nDual quantization error at epoch {self.epoch_cur}: {total_loss / len(data)}")

    def start_new_epoch(self, verbose=False):
        """Starts a new epoch."""
        self.epoch_cur += 1
        self.initialize_resource_map()
        if verbose:
            print(f"Starting epoch {self.epoch_cur}")

    def update(self, x, r, win, t, max_iteration):
        """Updates the weights of the neurons."""
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        sig = self._decay_function(self._sigma, t, max_iteration)
        g = self.neighborhood(win, sig) * eta
        self._weights += np.einsum('i,ij->ij', g, x - self._weights)

    def winner(self, x, rscore, return_distance=False):
        """Finds the winning neuron for a given input."""
        self._activate(x, rscore)
        winner_index = np.argmin(self._activation_map)
        self._resource_map[winner_index] -= 1
        if self._resource_map[winner_index] < 0:
            raise ValueError("Resource map is not updated correctly.")
        if return_distance:
            return winner_index, self._activation_map[winner_index]
        return winner_index

    def _activate(self, x, rscore):
        """Updates the activation map for a given input."""
        self._activation_map = self._activation_distance(x, rscore, self._weights, self._weights_radius)
        resource_map_mask = np.log(1 / (self._resource_map + 1e-11)) * 1e5
        self._activation_map += resource_map_mask

    def _euclidean_distance(self, x, r, w, wr):
        """Computes the combined distance between the input and the weights."""
        raw_dist = np.linalg.norm(x - w, axis=-1)
        normalized_dist = (raw_dist - np.min(raw_dist)) / (np.max(raw_dist) - np.min(raw_dist) + 1e-8)
        raw_radius_dist = np.exp((r - wr) ** 2)
        normalized_radius_dist = (raw_radius_dist - np.min(raw_radius_dist)) / (np.max(raw_radius_dist) - np.min(raw_radius_dist) + 1e-8)
        return self._ws * normalized_dist + self._wr * normalized_radius_dist

    def _check_iteration_number(self, num_iteration):
        """Checks if the number of iterations is valid."""
        if num_iteration < 1:
            raise ValueError('num_iteration must be greater than 1.')

    def _check_input_len(self, data):
        """Checks that the data has the correct dimensionality."""
        data_len = len(data[0])
        if self._input_len != data_len:
            raise ValueError(f'Received data with {data_len} features, expected {self._input_len}.')

    def _gaussian(self, c, sigma):
        """Returns a Gaussian neighborhood function centered at c."""
        d = 2 * sigma * sigma
        dist_sq = np.sum((self._grid - self._grid[c]) ** 2, axis=-1)
        return np.exp(-dist_sq / d)


def get_grid_position_som(som, data, relevance, ids_same_order, sort=True, split=False):
    """Gets the grid positions from the SOM for each data point.

    Parameters
    ----------
    som : CircularSOM
        The trained Circular SOM.

    data : array-like, shape = [n_samples, n_features]
        The input data.

    relevance : array-like, shape = [n_samples]
        Relevance scores for each data point.

    ids_same_order : array-like, shape = [n_samples]
        The identifiers for each data point, in the same order as 'data'.

    Returns
    -------
    pos_res : dict
        Dictionary mapping data point IDs to their positions on the SOM grid.
    """
    pos_res = {}
    radius = 0.3
    som.initialize_resource_map()
    modified_data = zip(data, relevance, ids_same_order)
    if sort:
        modified_data = sorted(modified_data, key=lambda t: t[1], reverse=True)
    for x, r, i in modified_data:
        w = som.winner(x, r)
        loc = som._grid[w]
        if not split:
            pos_res[i] = [loc[0], loc[1]]
        else:
            xloc = loc[0]
            yloc = loc[1]
            if xloc > 0:
                xloc += radius * 2
            else:
                xloc -= radius * 2
            if yloc > 0:
                yloc += radius * 2
            else:
                yloc -= radius * 2
            pos_res[i] = [xloc, yloc]
    return pos_res


def generate_rr_projection(data, relevance, metadata, num_of_epochs=1, w_s=0.2, w_r=0.8, step=8, verbose=True, sigma=1.5, learning_rate=0.7, activation_distance='euclidean', topology='circular', neighborhood_function='gaussian', random_seed=10):
    """Generates a relevance-preserving projection using Circular SOM. This function is mainly created for frontend system to generate all the necessary data for visualization. 

    Parameters
    ----------
    data : array-like, shape = [n_samples, n_features]
        The input data.

    relevance : array-like, shape = [n_samples]
        Relevance scores for each data point.

    metadata : array-like, shape = [n_samples]
        Metadata associated with each data point.

    Returns
    -------
    som : CircularSOM
        The trained Circular SOM.

    df : pandas.DataFrame
        DataFrame containing the data points and their positions.
    """
    from sklearn.cluster import KMeans

    data_size = data.shape[0]
    embedding_size = data.shape[1]
    num_of_layers = get_best_number_of_layers(step, data_size)
    _relevance = np.exp(relevance)
    relevance_normalized = (_relevance - np.min(_relevance)) / (np.max(_relevance) - np.min(_relevance))

    som = CircularSOM(step, num_of_layers, embedding_size, sigma=sigma, learning_rate=learning_rate, activation_distance=activation_distance, topology=topology, neighborhood_function=neighborhood_function, random_seed=random_seed)

    if verbose:
        print("Starting SOM training...")
    som.train(data, relevance_normalized, data_size * num_of_epochs, w_s=w_s, w_r=w_r, verbose=verbose, report_error=False, use_sorted=True)
    if verbose:
        print("SOM training completed.")

    circle_pos = get_grid_position_som(som, data, relevance_normalized, ids_same_order=np.arange(data_size))
    data_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=20).fit_transform(data)
    if verbose:
        print("Data processing completed.")

    data_dict = [{
        "pos_x": data_embedded[i][0],
        "pos_y": data_embedded[i][1],
        "circle_x": circle_pos[i][0],
        "circle_y": circle_pos[i][1],
        "embedding": data[i],
        "relevance": relevance_normalized[i],
        "metadata": metadata[i]
    } for i in range(data_size)]
    df = pd.DataFrame(data_dict)

    embeddings = np.array(df['embedding'].tolist())
    kmeans = KMeans(n_clusters=8, random_state=0, n_init="auto").fit(embeddings)
    df["category_num"] = kmeans.labels_

    return som, df


