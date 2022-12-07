#https://github.com/UKPLab/sentence-transformers

from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the data of the images and its prompts, without index
root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
data_path = os.path.join(root, 'data/images_prompts.csv')
images_prompts = pd.read_csv(data_path, sep=",", header=0, index_col = "filename")

# Get the embeddings of the prompts as a column in the dataframe
prompts_embeddings = model.encode(images_prompts['prompt'], show_progress_bar=True)
#images_prompts['prompt_embedding'] = prompts_embeddings.tolist()

# Get an input sentence from the user
input_sentence = input("Enter a sentence: ")
#k = int(input("Enter the number of images to retrieve: "))
k = 3

# Get the embedding of the input sentence
input_embedding = model.encode(input_sentence, show_progress_bar=True)

# Compute the cosine similarity between the input sentence and the prompts
cosine_similarities = [1 - cosine(input_embedding, prompt_embedding) for prompt_embedding in prompts_embeddings]

# Get the indices of the k most similar prompts
most_similar_indices = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:k]

# Print the most similar prompts
print("Most similar prompts:")
for index in most_similar_indices:
    print(images_prompts.iloc[index]['prompt'])

# Save as image the grid of the most similar images to the input sentence
images_folder = os.path.join(root, 'images/')
images_filenames = [os.path.join(images_folder, images_prompts.iloc[index].name) for index in most_similar_indices]
images = [plt.imread(image_filename) for image_filename in images_filenames]

fig = plt.figure(figsize=(10., 10.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(1, k),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.

                    )

for ax, im in zip(grid, images):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

# add the sentence as title
fig.suptitle(input_sentence, fontsize=16)

# add the prompts as subtitles
for i, index in enumerate(most_similar_indices):
    grid[i].set_title(images_prompts.iloc[index]['prompt'], fontsize=8)

# remove the axis
for ax in grid:
    ax.axis('off')

# save the figure
plt.savefig(os.path.join(root, 'output.png'))
plt.show()