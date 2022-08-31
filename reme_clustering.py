"""
7/08/2022 - Clément Dauvilliers
Parses the REME HTML file extracted from the PDF version, and tries to detect
p_elements using unsupervised clustering.
This file is made for practical application. The method was designed in
exploration.ipynb .
The HTML version of the PDF file was extracted using pdftohtml with the
complexe document option.
"""
import numpy as np
import re
import os
from bs4 import BeautifulSoup, NavigableString
from matplotlib import cm
from sklearn.cluster import AgglomerativeClustering

# Path to the pages of the HTML version
# This must be a directory in which each file is a specific
# page xxxx-{number of page}.html
# This is the output of pdftohtml if the -s option isn't used
HTML_PAGES_REP = os.path.join("docs", "html_pages")
# Directory of the new HTML pages over which the clusters have been drawed
NEW_HTML_REP = "visu_docs"
# Whether to draw the clusters on a new HTML file
DRAW_CLUSTERS = True

# Minimal distance in pixels between two clusters for them
# to be split during the clustering algorithm.
CLUSTERING_DISTANCE_THRESHOLD = 35


def get_pos(p_object):
    """
    Retrieves the position of a <p> element from its style attribute.
    Returns a pair (top, left) in pixels.
    """
    top, left = re.findall('[0-9]+', p_object.attrs['style'])
    return int(top), int(left)


def page_p_elements(p_objects, draw_clusters=False):
    """
    Detects the p_elements within a Soup corresponding
    to a single document page.
    :param p_objects: <p> elements of the soup within the page
        that is being processed.
    :param draw_clusters: Boolean, whether to add a background color
        to the p_elements to indicate the clusters.
    :return: a list of integers giving the predicted cluster
        for every <p> element.
    """
    # FIRST STEP: extract the locations on the page of every <p>
    locations = [get_pos(par) for par in p_objects]
    # Map (top, left) --> <p> object
    tops, lefts = [x for x, y in locations], [y for x, y in locations]

    # SECOND STEP: Clustering
    X = np.array([tops, lefts]).T
    pred = AgglomerativeClustering(linkage="single",
                                   distance_threshold=CLUSTERING_DISTANCE_THRESHOLD,
                                   n_clusters=None,
                                   affinity="l1").fit_predict(X)

    # THIRD STEP (OPTIONAL): Draw the clusters over a new HTML document
    if draw_clusters:
        n_clusters = pred.max() + 1
        clusters = [[] for _ in range(n_clusters)]
        for i, prediction in enumerate(pred):
            clusters[prediction].append(p_objects[i])
        cmap = cm.get_cmap('tab20')
        for k, cluster in enumerate(clusters):
            # Assign a color to the cluster
            color = cmap(k / n_clusters, bytes=True)
            # Assign that color as background to all elements in that cluster
            for elem in cluster:
                elem['style'] += ";background-color:rgba({}, {}, {}, 0.3);".format(*color)

    return pred


if __name__ == "__main__":
    # Apply the clustering to every page
    for k, page_path in enumerate(sorted(os.listdir(HTML_PAGES_REP))):
        print(f"Reading HTML file {page_path}")
        with open(os.path.join(HTML_PAGES_REP, page_path), "r") as page_html:
            page = page_html.read()
        soup = BeautifulSoup(page, features="html.parser")
        print("Done")
        print("Parsing...")

        # Removes all tags in the page empty of any human-readable text
        for tag in soup.find_all():
            if len(tag.get_text(strip=True)) == 0:
                tag.extract()

        # Retrieves all <p> elements in the page
        p_elements = soup.body.find('div').find_all('p', recursive=False)

        # The <p> tags do not always directly contain their text, it may be included
        # within children tags (such as <b> for bold words).
        # It will be easier to work with text that is directly contained in par.string, so
        # we'll set the value of that attribute to the value of get_text(), which retrieves
        # all the text recursively in the tag's children.
        for par in p_elements:
            # Retrieves all of the text contained in the <p> tag
            string = par.get_text()
            # Removes all children of the <p> tag
            for child in par.children:
                child.extract()
            # Sets the same text as before as raw text within the <p>
            par.string = string

        # Treating the problem of "1er degré", as on page 16:
        # The "er", "nd" and "eme" characters create a separate paragraph
        # which destroys the clustering

        # We'll add to new_list all <p> objects that should be kept, as we're going
        # to merge together so of them
        # We'll use a manual iterator instead of a for loop as we want to skip
        # elements during the loop
        par_iter = iter(p_elements)
        # During the loop, will always be the last <p> tag that has been kept.
        prev_par = None
        # List that will contain the p_elements that have been kept
        new_list = []

        while True:
            try:
                par = next(par_iter)
                text = par.get_text()
                if text in ['er', 'nd', 'eme']:
                    # Adds to the previous paragraph the content of this one and the next one too
                    # Ex: "Assistance éducative 1" + "er" + "degré"
                    next_par = next(par_iter)
                    prev_par.string = prev_par.get_text() + text + " " + next_par.get_text()
                    # Removes the current and next <p> from the page as their content is now in
                    # the previous <p>
                    par.extract()
                    next_par.extract()
                else:
                    # Adds the paragraph to the list of kept p_elements
                    new_list.append(par)
                    # Updates the previous <p> tag
                    prev_par = par
            except StopIteration:
                break
        # Update the list of <p> tags to only the ones that have been kept
        p_elements = new_list

        # Cleaning the text
        # -- Replaces all &nbsp characters with traditional whitespaces
        for par in p_elements:
            par.string = par.string.replace(u'\xa0', ' ')

        # Performs the clustering
        clusters = page_p_elements(p_elements, draw_clusters=DRAW_CLUSTERS)

        # Save the new HTML file over which the clusters have been drawed
        with open(os.path.join(NEW_HTML_REP, page_path), "w") as file:
            file.write(str(soup))
    print("Done")
