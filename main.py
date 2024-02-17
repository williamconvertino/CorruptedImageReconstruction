import random
import math
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

import scipy as sp

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ========================== Constants & Image Initialization ===================================================

image_1 = plt.imread('fishing_boat.bmp').astype(np.float32)
image_2 = plt.imread('nature.bmp').astype(np.float32)

# ========================== Visualization ===================================================

window_id = 1

nan_map = plt.colormaps['gray']
nan_map.set_bad(color='orange')

def show_image(image, title="", reshape = False, mutli_figure = False):

    if reshape:
        image = image.reshape(reshape)
    
    if mutli_figure:
        global window_id
        window_id += 1
        plt.figure(window_id, figsize=(10, 6))
    else:
        plt.figure(figsize=(10, 6))

    plt.imshow(image, cmap=nan_map)
    plt.title(title)
    plt.colorbar()

    # plt.imsave('corrupted_image.bmp', image, cmap=nan_map)

    if not mutli_figure:
        plt.show()

def show_image_grid(images, titles=[], reshape = False, mutli_figure = False, header = ""):

    nan_map = plt.colormaps['gray']
    nan_map.set_bad(color='orange')
    
    if mutli_figure:
        global window_id
        window_id += 1
        fig = plt.figure(window_id, figsize=(10, 6))
    else:
        fig = plt.figure(figsize=(10, 6))

    fig.suptitle(header)

    num_cols = int(math.ceil(len(images)**0.5))
    num_rows = (len(images) + num_cols - 1) // num_cols

    for i in range(len(images)):
        image = images[i]
        title = titles[i] if len(titles) > i else ""

        if reshape:
            image = image.reshape(reshape)
            
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(image, cmap=nan_map)
        plt.title(title)
        plt.colorbar()

    plt.tight_layout()
    
    if not mutli_figure:
        plt.show()

def plot_weights(weights, title="", mutli_figure = False):

    if mutli_figure:
        global window_id
        window_id += 1
        plt.figure(window_id, figsize=(10, 6))
    else:
        plt.figure(figsize=(10, 6))

    plt.stem(weights)
    plt.title(title)

    if not mutli_figure:
        plt.show()

def plot_MSEs(MSEs_1, MSEs_2, title="", mutli_figure = False):

    if mutli_figure:
        global window_id
        window_id += 1
        plt.figure(window_id, figsize=(10, 6))
    else:
        plt.figure(figsize=(10, 6))

    S_values_1 = [item[0] for item in MSEs_1]
    MSE_values_1 = [item[1] for item in MSEs_1]

    S_values_2 = [item[0] for item in MSEs_2]
    MSE_values_2 = [item[1] for item in MSEs_2]

    plt.plot(S_values_1, MSE_values_1, color='red')
    plt.plot(S_values_2, MSE_values_2, color='green')
    plt.title(title)
    plt.xlabel('S')
    plt.ylabel('MSE')

    if not mutli_figure:
        plt.show()

# ========================== Chip Generation ===================================================

def get_chip_vector(image, width, height, offset_x=0, offset_y=0, S=None):
    
    top, bottom, left, right = offset_y, offset_y + height, offset_x, offset_x + width
    
    chip = image[top:bottom, left:right].flatten()

    if S is not None:
        chip = corrupt_chip(chip, S)
    
    return chip

def chip_mse(chip_1, chip_2):
    return np.mean((chip_1 - chip_2) ** 2)

# ========================== Image Corruption ===================================================

def corrupt_chip(image, S):

    if len(image.shape) == 1:
        corrupt_image = image.copy()
    else:
        corrupt_image = image.flatten()
    
    num_total_pixels = corrupt_image.size
    num_corrupted_pixels = num_total_pixels - S

    corrupted_pixel_indices = pd.Series(corrupt_image).sample(num_corrupted_pixels)

    corrupt_image[corrupted_pixel_indices.index] = np.nan

    return corrupt_image.reshape(image.shape)

def corrupt_image(image, chip_size, S):

    corrupted_image = image.copy()
    
    width = image.shape[1]
    height = image.shape[0]
    total_pixels = width * height
    
    for x in range(0, width, chip_size):
        for y in range(0, height, chip_size):
            corrupted_chip = get_chip_vector(image, chip_size, chip_size, x, y, S).reshape(chip_size, chip_size)
            corrupted_image[y : y + chip_size, x : x + chip_size] = corrupted_chip

    return corrupted_image

# ========================== Basis Chip Generation ===================================================

def get_basis_value(x, y, u, v, Q, P):
    
    x = x + 1
    y = y + 1
    # u = u + 1
    # v = v + 1

    alpha = np.sqrt(1/P) if u == 0 else np.sqrt(2/P)
    beta = np.sqrt(1/Q) if v == 0 else np.sqrt(2/Q)
    
    return alpha * beta * np.cos((np.pi * (2*x - 1) * u) / (2 * P)) * np.cos((np.pi * (2*y - 1) * v) / (2 * Q))

def get_basis_chip(u, v, P, Q):
    
    image = np.zeros((Q, P))
    
    for x in range(P):
        for y in range(Q):
            image[y][x] = get_basis_value(x, y, u, v, Q, P)

    return image

def get_basis_matrix(P, Q):

    basis_matrix = pd.DataFrame(np.zeros((P * Q, P * Q)))
    
    for v in range(Q):
        for u in range(P):
            basis_matrix.iloc[:, v * Q + u] = pd.Series(get_basis_chip(u, v, P, Q).flatten())
    
    return basis_matrix

# ========================== Lasso ===================================================

def get_log_spaced_regularization_parameters(x_min, x_max, candidates_per_decade):
    
    decades = [10**i for i in range(x_min, x_max + 1)]
    
    regularization_parameters = []

    for i in range(len(decades)-1):
        for j in range(candidates_per_decade):
            regularization_parameters.append(random.uniform(decades[i], decades[i+1]))
    
    return regularization_parameters

def run_lasso():
    pass


# ========================== Cross Validation ===================================================

def generate_chip_with_validation(image, width, height, offset_x, offset_y, S=None):
    
    chip_vector = get_chip_vector(image, width, height, offset_x, offset_y, S)

    sensed_pixels = pd.Series(chip_vector).dropna()
    corrupted_pixels = pd.Series(chip_vector).isna()

    basis_matrix = get_basis_matrix(8, 8)
    sensed_basis_matrix = basis_matrix.loc[sensed_pixels.index]
    corrupted_basis_matrix = basis_matrix.loc[corrupted_pixels.index]

    reg_params = get_log_spaced_regularization_parameters(-3, 7, candidates_per_decade=2)

    num_sensed = sensed_pixels.size

    m = math.floor(num_sensed/6)
    num_subsets = 20

    MSEs = pd.DataFrame(index=reg_params, columns=range(num_subsets))

    for i in range(num_subsets):

        test_set = sensed_pixels.sample(n=m)
        training_set = sensed_pixels.drop(test_set.index)

        training_basis_matrix = basis_matrix.loc[training_set.index]
        testing_basis_matrix = basis_matrix.loc[test_set.index]

        subset_MSEs = []

        for alpha in reg_params:
            lasso = Lasso(alpha=alpha)
            lasso.fit(training_basis_matrix, training_set)

            # predicted_values = sensed_basis_matrix @ lasso.coef_ + lasso.intercept_
            
            # reconstructed_chip_vector = chip_vector.copy()
            # reconstructed_chip_vector[sensed_pixels.index] = predicted_values[sensed_pixels.index]
            # reconstructed_chip_vector[test_set.index] = predicted_values[test_set.index]

            # 2.a
            # show_image_grid([chip_vector, reconstructed_chip_vector], ["Original", f"Reconstructed with alpha={round(alpha, 4)}"], reshape=(8, 8), mutli_figure=True)
            # plot_weights(lasso.coef_, title=f"Weights for alpha {round(alpha, 4)}", mutli_figure=True)

            predicted_values = testing_basis_matrix @ lasso.coef_ + lasso.intercept_

            mse = chip_mse(test_set, predicted_values)
            subset_MSEs.append(mse)

        MSEs[i] = subset_MSEs

    MSEs.sort_index(inplace=True)

    average_MSEs = MSEs.mean(axis=1)

    best_alpha = average_MSEs.idxmin()

    lasso = Lasso(alpha=best_alpha)
    lasso.fit(sensed_basis_matrix, sensed_pixels)
    
    predicted_values = corrupted_basis_matrix @ lasso.coef_ + lasso.intercept_
    reconstructed_chip_vector = chip_vector.copy()
    reconstructed_chip_vector[corrupted_pixels.index] = predicted_values[corrupted_pixels.index]

    # show_image_grid([chip_vector, reconstructed_chip_vector], ["Original", f"Reconstruction"], reshape=(8, 8), mutli_figure=True, header=f"Best reconstruction with alpha={round(best_alpha, 4)}, predicted MSE={round(average_MSEs[best_alpha], 4)}")
    # plot_weights(lasso.coef_, title=f"Weights for alpha {round(best_alpha, 4)}", mutli_figure=True)

    return reconstructed_chip_vector.reshape(height, width)

# ========================== Image Reconstruction ===================================================

def fix_corruption(image):

    print("Starting Reconstruction...")

    reconstructed_image = image.copy()

    width = image.shape[1]
    height = image.shape[0]
    total_pixels = width * height

    chip_size = 8

    num_chips = total_pixels / (chip_size ** 2)
    current_chip = 0

    start_time = time.time()

    for x in range(0, width, chip_size):
        for y in range(0, height, chip_size):
            reconstructed_chip = generate_chip_with_validation(image, chip_size, chip_size, x, y)
            reconstructed_image[y : y + chip_size, x : x + chip_size] = reconstructed_chip

            if current_chip % 50 == 0:
                print(f"======== {current_chip}/{num_chips} chips reconstructed ({current_chip / num_chips}%) ========")
                
                elapsed_time = time.time() - start_time
                
                remaining_time = (elapsed_time * (current_chip / num_chips))/(1 - (current_chip / num_chips))

                print(f"\t\tElapsed time: {round(elapsed_time / 60)} minutes")
                print(f"\t\tEstimated time remaining: {round(remaining_time / 60)} minutes")

            current_chip += 1

    print("Reconstruction Complete")

    # plt.imsave('reconstructed_image.bmp', reconstructed_image, cmap=nan_map)

    return reconstructed_image

# ========================== Testing ===================================================

def TestReconstruction():

    image = image_1

    reconstructed_MSEs = []
    med_3_MSEs = []

    for S in [10, 20, 30]:
        name = f'FishingBoat_S={S}'
        corrupted_image = corrupt_image(image, 8, S)

        reconstructed_image = fix_corruption(corrupted_image)
        # reconstructed_image = plt.imread('{name}_Reconstructed.bmp').astype(np.float32)[:, :, 0]
        reconstructed_MSE = chip_mse(image, reconstructed_image)
        plt.imsave(f'{name}_Reconstructed.bmp', reconstructed_image, cmap=nan_map)
        
        med_3_image = sp.signal.medfilt2d(reconstructed_image, kernel_size=3)
        # reconstructed_image = plt.imread('{name}_Filtered.bmp').astype(np.float32)[:, :, 0]
        med_3_MSE = chip_mse(image, med_3_image)
        plt.imsave(f'{name}_Filtered.bmp', med_3_image, cmap=nan_map)
        
        reconstructed_MSEs.append((S,reconstructed_MSE))
        med_3_MSEs.append((S, med_3_MSE))

        show_image_grid([image, corrupted_image, reconstructed_image, med_3_image], ["Original", "Corrupted", f"Reconstructed, MSE={reconstructed_MSE}", f"Filtered, MSE={med_3_MSE}"], mutli_figure=True, header=f'Reconstruction with S={S}')

    plot_MSEs(reconstructed_MSEs, med_3_MSEs, title="Unfiltered vs Filtered MSEs", mutli_figure=True)


# ========================== Main ===================================================
def main():
    TestReconstruction()

if __name__ == "__main__":
    main()
    plt.show()