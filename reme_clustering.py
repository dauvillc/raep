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
import pandas as pd
from collections import defaultdict
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
# Path into which the data is saved
SAVE_PATH = os.path.join("docs", "reme_data.json")
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
    top, left = re.findall('[0-9]+px', p_object.attrs['style'])
    # "top" and "left" are now strings such as "1024px", "681px", but we want to
    # ignore the "px" part
    return int(top[:-2]), int(left[:-2])


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

    return clusters


if __name__ == "__main__":
    # List that will contain the data for each page
    pages_data = []
    # We process the document page per page
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
        # -- Converts the text to lowercase
        for par in p_elements:
            par.string = par.string.lower()

        # Performs the clustering
        clusters = page_p_elements(p_elements, draw_clusters=DRAW_CLUSTERS)

        # If less than 5 clusters have been detected, then it's not a job sheet page
        # but a new chapter title page. We just skip it.
        if len(clusters) < 5:
            continue

        # Save the new HTML file over which the clusters have been drawed
        with open(os.path.join(NEW_HTML_REP, page_path), "w") as file:
            file.write(str(soup))

        # =========== SECOND PART: PROCESSING THE CLUSTERS ======================= #
        # ------------ BUILDING A SPATIAL REPRESENTATION ------------------------- #
        # We'll need to order the clusters list by ascending top coordinate. The top
        # coordinate of a cluster is that of the top-most paragraph within it.
        cluster_coords = []
        for cluster in clusters:
            par_coords = [get_pos(par) for par in cluster]
            top = min((x for x, y in par_coords))
            left = min((y for x, y in par_coords))
            cluster_coords.append((top, left))

        sorted_clusters_and_coords = [(cluster, top, left) for cluster, (top, left) in zip(clusters, cluster_coords)]
        # Sorts the list (cluster, top, left) by the 'top' element in ascending order
        sorted_clusters_and_coords = sorted(sorted_clusters_and_coords,
                                            key=lambda cluster_and_coords: cluster_and_coords[1])
        # Rebuilds the clusters and cluster_coords list, this time in the right order
        clusters = [cluster for cluster, _, _ in sorted_clusters_and_coords]
        cluster_coords = [(top, left) for _, top, left in sorted_clusters_and_coords]

        # ---------- IDENTIFYING SECTIONS ----------------------------------------- #
        # This is the part where we associate each cluster with a known section of the sheets
        # We begin by reading a file that contains the names of every section we could find
        with open('docs/sections.txt', 'r') as sections_file:
            sections = [line.strip() for line in sections_file]
        # map Section name --> cluster
        sections_dict = dict()

        # ---- Identifying by occurence of the section name ----- #
        # Here we try to find the names of the sections within the clusters' text.
        for cluster in clusters:
            for par in cluster:
                text = par.get_text()
                for section_name in sections:
                    if section_name in text:
                        sections_dict[section_name] = cluster
                        # Removes the section name from the text of the cluster
                        par.string = par.string.replace(section_name, '')
                        # Note: the <p> tag still exists in the cluster, so its location will
                        # be taken into account to compute the location of the cluster in the
                        # next part of the process

        # ---- Identifying by location on the page ----- #
        sections_dict['chapitre'] = clusters[0]
        sections_dict['nom du métier'] = clusters[1]
        sections_dict['description'] = clusters[3]

        # ---- 'Activités principales' section ---- #
        if 'activités principales' in sections_dict:
            # Browses the clusters list (which is still ordered by ascending top coordinate)
            # until finding the title "activités principales"
            for k, cluster in enumerate(clusters):
                if cluster == sections_dict['activités principales']:
                    activ_title_cluster = k
                    break
            # Merges the two content clusters into one
            sections_dict['activités principales'] = clusters[k + 1] + clusters[k + 2]

        # ---- Merging the text of paragraphs ---- #
        sections_text = dict()
        for section_name, cluster in sections_dict.items():
            cluster_text = ""
            for par in cluster:
                cluster_text += par.string + ' '
            sections_text[section_name] = cluster_text

        # ----------- Extracting the list items -------------- #
        # Map to automatically retrieve the right separator for a given
        # section
        separator = defaultdict(lambda: '•')
        separator['activités principales'] = '◗'
        separator['correspondances statutaires'] = ','

        # Converts the text of every section into a list
        section_lists = dict()
        for section_name, section_text in sections_text.items():
            # Cleaning:
            # We'll remove some undesirable special characters
            for char in ['➜', '[', ']']:
                section_text = section_text.replace(char, '')
            # We'll change the double spaces to simple whitespaces
            section_text = section_text.replace('  ', ' ')

            items = section_text.split(separator[section_name])
            # Removes empty list items
            items = [item.lstrip().strip() for item in items]
            items = [item for item in items if len(item) > 0]
            section_lists[section_name] = items

        pages_data.append(section_lists)

    # Converts to a pandas dataframe
    df = pd.DataFrame(pages_data)
    # Some sections shouldn't be lists but rather single strings
    nonlist_sections = ['code fiche', 'description', 'nom du métier', 'chapitre']
    df[nonlist_sections] = df[nonlist_sections].applymap(lambda items: items[0])
    # Merges the data from the first and second pages that correspond to the same job
    df = df.groupby('code fiche').first().reset_index()
    df.to_json(SAVE_PATH)
    print("Done, saved to ", SAVE_PATH)
