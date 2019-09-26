# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
import torch


COLORS = ["salmon", "skyblue", "lightgreen", "plum", 
          "wheat", "lightpink", "aquamarine", "lightsteelblue"]


def save_features(feature_list, save_dir, mode="wb"):
    """Save a list of features to file.

    feature_shape: 
        (num_samples, feature_dim1, feature_dim2, ...)
    """
    num_features = len(feature_list)
    num_samples = feature_list[0].shape[0]
    flatten_features = [
        feas.reshape(num_samples, -1) for feas in feature_list]
    with open(save_dir, mode=mode) as f:
        pickle.dump(flatten_features, f)


def load_features(load_dir, max_samples=10**4):
    """Load lists of features from file.

    feature_shape: (num_samples, feature_dim)
    """
    res = {}
    cnt = 0
    with open(load_dir, "rb") as f:
        while cnt < max_samples:
            try:
                feature_list = pickle.load(f)
                for i, feas in enumerate(feature_list):
                    if isinstance(feas, torch.Tensor):
                        feas = feas.cpu().detach().numpy()
                    if i in res:
                        res[i].append(feas)
                    else:
                        res.update({i: [feas]})
                cnt += res[0][-1].shape[0]
            except EOFError:
                break
    feature_list = [np.concatenate(res[i]) for i in res]
    return feature_list


def compute_correlation(covariance):
    variance = np.diag(covariance).reshape(-1, 1)
    stds = np.sqrt(np.matmul(variance, variance.T))
    correlation = covariance / (stds + 1e-16)
    return correlation


def compute_covariance_offline(feature_list):
    """Compute covariance matrix for high-dimensional features.

    feature_shape: (num_samples, feature_dim)
    """
    num_features = len(feature_list)
    num_samples = feature_list[0].shape[0]
    flatten_features = [
        feas.reshape(num_samples, -1) for feas in feature_list]
    unbiased_features = [
        feas - np.mean(feas, 0) for feas in flatten_features]
    # (num_samples, feature_dim, num_features)
    features = np.stack(unbiased_features, -1)
    covariance = np.zeros((num_features, num_features), np.float32)
    for i in range(num_samples):
        covariance += np.matmul(features[i].T, features[i])
    return covariance


def compute_covariance_online(feature_list,
                              num_samples,
                              mean_list,
                              covariance_list):
    """Compute covariance matrix for high-dimensional features, 
    update means and covariances in-place.

    feature_shape: (feature_dim,)
    """
    num_features = len(feature_list)
    num_samples += 1
    flatten_features = [feas.flatten() for feas in feature_list]
    for i in range(num_features):
        tmp0 = flatten_features[i] - mean_list[i]
        mean_list[i] += tmp0 / num_samples
        tmp1 = flatten_features[i] - mean_list[i]
        covariance_list[i][0] += np.matmul(tmp0, tmp1) / num_samples
        for j in range(i, num_features):
            covariance_list[i][j - i] += \
                np.matmul(flatten_features[j] - mean_list[j], tmp1)

    return num_samples


def pca_projection(features, k):
    num_samples = features.shape[0]
    flatten_features = features.reshape(num_samples, -1)
    unbiased_features = flatten_features - np.mean(flatten_features, 0)
    pca = PCA(n_components=k)
    pca_fit = pca.fit(unbiased_features)
    eigen_vectors = pca_fit.components_
    explained_variance = pca_fit.explained_variance_
    projects = pca.transform(unbiased_features)
    return eigen_vectors, explained_variance, projects


def plot_clusters(feature_list, k, save_dir, **kwargs):
    """Cluster features into groups and save.

    feature_shape: (feature_dim,)
    """
    num_features = len(feature_list)
    flatten_features = [feas.flatten() for feas in feature_list]

    kmeans = KMeans(n_clusters=k)
    kmeans_fit = kmeans.fit(flatten_features)
    labels = kmeans_fit.labels_.tolist()
    cluster_centers = kmeans_fit.cluster_centers_

    features = np.stack(flatten_features + [c for c in cluster_centers])
    unbiased_features = features - np.mean(features, 0)
    pca = PCA(n_components=2)
    pca_fit = pca.fit(unbiased_features[:-k])
    projects = pca_fit.transform(unbiased_features)

    feature_colors = [COLORS[labels[i]] for i in range(num_features)]
    cluster_colors = [COLORS[i] for i in range(k)]
    feature_annotates = kwargs.get("annotate", [])
    cluster_annotates = []

    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(projects[:-k, 0], projects[:-k, 1], 
        s=100, c=feature_colors, marker="o", alpha=0.8)
    # ax.scatter(projects[-k:, 0], projects[-k:, 1], 
    #     s=160, c="none", edgecolor=cluster_colors, marker="^", alpha=1.0)
    for pt, annot in zip(projects, feature_annotates):
        ax.annotate(annot, pt)
    ax.set_title("feature clusters", fontsize=10)
    plt.savefig(save_dir, bbox_inches="tight")
    plt.close()


def plot_confidence(feature_list, save_dir, **kwargs):
    """Plot confidence ellipses of a list of features.

    feature_shape: (num_samples, feature_dim)
    """
    num_features = len(feature_list)
    annotates = kwargs.get("annotate", [""] * num_features)

    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(num_features):
        confidence_ellipse(features=feature_list[i],
                           ax=ax,
                           n_std=1.0,
                           edgecolor=COLORS[i],
                           facecolor="none",
                           linewidth=1.5,
                           annotate=annotates[i])

    ax.set_aspect("equal", adjustable="box")
    ax.set_title("confidence ellipse", fontsize=10)
    plt.autoscale(tight=True)
    plt.savefig(save_dir, bbox_inches="tight")
    plt.close()


def confidence_ellipse(features, ax, n_std=1.0, **kwargs):
    """Draw a confidence ellipse for samples of multiple random variables.

    feature_shape: (num_samples, num_features)
    """
    annotate = kwargs.pop("annotate", "")
    mean = np.mean(features, 0)
    unbiased_features = features - mean

    pca = PCA(n_components=2)
    pca_fit = pca.fit(unbiased_features)
    eigen_vectors = pca_fit.components_
    explained_variance = pca_fit.explained_variance_

    center = pca_fit.transform([mean])[0]
    radius = pca_fit.transform(eigen_vectors)
    angle = np.arccos(radius[0][0] / np.linalg.norm(radius[0])) * 180.0 / np.pi

    ax.plot(*center, "o", markersize=6, markeredgewidth=1.5, 
            markeredgecolor=kwargs["edgecolor"], markerfacecolor="none")
    ellipse = Ellipse(xy=center,
                      width=explained_variance[0] * n_std * 2,
                      height=explained_variance[1] * n_std * 2,
                      angle=angle,
                      **kwargs)
    ax.add_patch(ellipse)
    ax.annotate(annotate, center)
