"""
7/08/2022 - Clément Dauvilliers
Parses the REME HTML file extracted from the PDF version, and tries to detect
paragraphs using unsupervised clustering.
This file is made for practical application. The method was designed in
exploration.ipynb .
The HTML version of the PDF file was extracted using pdftohtml with the
complexe document option.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import os
from copy import deepcopy
from bs4 import BeautifulSoup
from unicodedata import normalize as uni_normalize
from matplotlib import cm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix

# Path to the full HTML version of the REME pdf.
HTML_FILE = os.path.join("docs", "rep_metiers.html")
# Minimal distance in pixels between two clusters for them
# to be split during the clustering algorithm.
CLUSTERING_DISTANCE_THRESHOLD = 37.5


def get_pos(p_object):
    """
    Retrieves the position of a <p> element from its style attribute.
    Returns a pair (top, left) in pixels.
    """
    top, left = re.findall('[0-9]+', p_object.attrs['style'])
    return int(top), int(left)


def page_paragraphs(p_objects):
    """
    Detects the paragraphs within a Soup corresponding
    to a single document page.
    :param p_objects: <p> elements of the soup within the page
        that is being processed.
    :return: a list of integers giving the predicted cluster
        for every <p> element.
    """
    # FIRST STEP: extract the locations on the page of every <p>
    # Map (top, left) --> <p> object
    par_dict = {get_pos(par): par for par in p_objects}
    # List of all (top, left) locations
    locations = list(par_dict.keys())
    tops, lefts = [x for x, y in locations], [y for x, y in locations]

    # SECOND STEP: Clustering
    X = np.array([tops, lefts]).T
    pred = AgglomerativeClustering(linkage="single",
                                   distance_threshold=CLUSTERING_DISTANCE_THRESHOLD,
                                   n_clusters=None,
                                   affinity="l1").fit_predict(X)

    # THIRD STEP (OPTIONAL): Draw the clusters over a new HTML document
    n_clusters = pred.max() + 1
    clusters = [[] for _ in range(n_clusters)]
    for i, prediction in enumerate(pred):
        clusters[prediction].append(p_objects[i])


if __name__ == "__main__":
    with open(HTML_FILE, "r") as reme_html:
        full_html = reme_html.read()
    soup = BeautifulSoup(full_html)
