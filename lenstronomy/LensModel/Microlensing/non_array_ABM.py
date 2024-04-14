# Original code fom Sameer Singh: https://github.com/Singh2202/2023/blob/main/Comprehensive%20notebook%20-%20latest.ipynb

import itertools
import matplotlib.pyplot as plt
import numpy as np
import math

def sub_pixel_creator(center, side_length, n_p):
    
    """Takes square centered at center = (x, y) with side_length = side length 
       (float) and divides it into n_p (integer) squares along each axis - so 
       returns n_p * n_p subsquares from the original square. In particular, 
       the function returns the coordinates of the centers of these subsquares,
       along with the final side length of the subsquare."""
     
    step_size = side_length / n_p
    leftmost_x = center[0] - side_length / 2
    lowest_y = center[1] - side_length / 2

    center_xs, center_ys = [], []
    center_x, center_y = leftmost_x + step_size / 2, lowest_y + step_size / 2
    for i in range(n_p):
        center_xs.append(center_x)
        center_ys.append(center_y)
        center_x += step_size
        center_y += step_size
    
    centers = list(itertools.product(center_xs, center_ys))
    new_side_length = step_size 
    return centers, new_side_length

# define the microlens

d_l = 4000  # distance of the lens in pc
d_s = 8000  # distance of the source in pc
M0 = 0.01 # mass of the lens in units of M_sol
diameter_s = 20 # size of the diameter of the source star in units of the solar radius

# compute lensing properties

from lenstronomy.Cosmo.micro_lensing import einstein_radius, source_size
theta_E = einstein_radius(M0, d_l, d_s)
size_s = source_size(diameter_s, d_s)

# compute ray-tracing grid

grid_scale = size_s / 80
grid_width = theta_E * 4
num_pix = int(grid_width / grid_scale)

from lenstronomy.Util import util
x, y = util.make_grid(numPix=num_pix, deltapix=grid_scale)

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

# compute unlensed surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': 0, 'center_x': 0, 'center_y': 0}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
unlensed_flux = np.sum(surface_brightness)


# compute surface brightness
lens = LensModel(lens_model_list=['POINT_MASS'])
kwargs_lens = [{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]
beta_x, beta_y = lens.ray_shooting(x, y, kwargs=kwargs_lens)
ligth = LightModel(light_model_list=['ELLIPSOID'])
kwargs_light = [{'amp': 1, 'radius': size_s/2, 'e1': 0, 'e2': 0, 'center_x': 0, 'center_y': 0}]
surface_brightness = ligth.surface_brightness(beta_x, beta_y, kwargs_light)
lensed_flux = np.sum(surface_brightness)

reference_magnification = lensed_flux / unlensed_flux
print(reference_magnification)

image = util.array2image(surface_brightness)
plt.imshow(image)
plt.colorbar()
plt.show()

L = theta_E * 4 # side length of square area in image plane - same as lenstronomy grid width
beta_0 = 4 * L # initial search radius - few times bigger than "necessary" to be safe (delta_beta)
beta_s = size_s / 2 # factor of 1/2 because radius
n_p = 30
eta = 0.7 * n_p
source_position = (0, 0)

# helper functions 

#defines loop_information to defines number of iterations and final scale factor  
def loop_information(eta, beta_0, beta_s):
    
    N = 1 + math.log((beta_0 / beta_s), eta)
    number_of_iterations = math.ceil(N)
    N_star = N - math.floor(N)
    final_eta = eta ** N_star 

    return number_of_iterations, final_eta

#creates an loop_info array to store number of iterations and final scale factor
loop_info = loop_information(eta, beta_0, beta_s)
number_of_iterations = loop_info[0]
final_eta = loop_info[1]

#defines within_distance to calculate distance and make sure that source polane boundary is within the source radius
def within_distance(center_point, test_point, threshold):
    
    distance = ((center_point[0] - test_point[0]) ** 2 + (center_point[1] - test_point[1]) ** 2) ** (1 / 2)
    if distance < threshold:
        return True
    else: 
        return False

# Defines the adaptive boundary mesh function
def ABM_non_array(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta, kwargs_lens):

    """
    Returns list of those high resolution image-plane pixels that were 
    mapped to within the radius β_s around the source position (β1, β2) 
    in the source plane.

    :param source_position: tuple; Coordinates of the source position (x, y). Source position
    :param L: float; Side length of square area in image plane. Same as lenstronomy grid width
    :param beta_0: float; Initial search radius (delta_beta)
    :param beta_s: float; Factor of 1/2 because radius
    :param n_p: int; Number of pixels
    :param eta: float; 0.7 * n_p
    :param number_of_iterations: int; Number of iterations
    :param final_eta: float; Final scale factor
    :param kwargs_lens: dict; Keyword arguments for lens model
    return: subset_centers: nparray; List of high resolution image-plane pixels that were mapped to within the radius β_s around the source position (β1, β2) in the source plane
    return: side_length: nparray; updated side length of square area in image plane
    return: int; total_number_of_rays_shot: total number of rays shot
    """
    
    # Initialize variables
    total_number_of_rays_shot = 0  # Counter for total number of rays shot
    i = 1  # Iteration counter
    centers = [(0, 0)]  # Initial center coordinates
    side_length = L  # Initial side length of square region (for source image)
    delta_beta = beta_0  # Initial step size for source plane radius
    
    # Main loop for adaptive boundary mesh algorithm
    while i < number_of_iterations:
        # Iterate through each center
        running_list_of_new_centers = []
        for center in centers:
            # Ray shoot from image to source plane and check if within distance
            source_coords = lens.ray_shooting(center[0], center[1], kwargs=kwargs_lens)
            total_number_of_rays_shot += 1
            if within_distance(source_coords, source_position, delta_beta):
                # Create sub-pixels and add to running list of centers (saying that if within distance create another iteration of pixels)
                resultant_centers = sub_pixel_creator(center, side_length, n_p)[0]
                running_list_of_new_centers.extend(resultant_centers)
        
        # Update centers and side length (/= opertaor divides the value of the variable on the left-hand side by the value on the right-hand side and assigns the result back to the variable on the left-hand side.)
        centers = running_list_of_new_centers
        side_length /= n_p
        
        # Update delta_beta based on iteration number
        if i < number_of_iterations:
            delta_beta /= eta
        elif i == number_of_iterations:
            delta_beta /= final_eta
            
        # Increment iteration counter
        i += 1
    
    # Find final centers within beta_s radius around source position
    final_centers = []
    for center in centers:
        source_coords = lens.ray_shooting(center[0], center[1], kwargs=kwargs_lens)
        total_number_of_rays_shot += 1
        if within_distance(source_coords, source_position, beta_s):
            final_centers.append(center)
    
    return final_centers, side_length, total_number_of_rays_shot

# Call the ABM function to compute high resolution pixels
high_resolution_pixels = ABM_non_array(source_position, L, beta_0, beta_s, n_p, eta, number_of_iterations, final_eta,[{'theta_E': theta_E, 'center_x': theta_E / 4, 'center_y': theta_E / 6}]) #last parameters are kwargs_lens parameters

# Compute the number of high resolution pixels
n = len(high_resolution_pixels[0]) 

# Compute the magnification using ABM results
computed_magnification = (n * high_resolution_pixels[1] ** 2) / (math.pi * (beta_s ** 2)) 

# Plot the high resolution pixels on a scatter plot
plt.scatter(*zip(*high_resolution_pixels[0]))
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.ylim(-2 * theta_E, 2 * theta_E)
plt.xlim(-2 * theta_E, 2 * theta_E)
plt.xlabel("Projected horizontal position on sky (arcseconds)")
plt.ylabel("Projected vertical position on sky (arcseconds)")
ax.invert_yaxis()

# Display lensed image
plt.figure()
plt.imshow(image)

# Print computed magnification and computation information
print("The ABM magnification is " + str(computed_magnification) + "; the lenstronomy magnification is " + str(reference_magnification))
print("ABM required " + str(high_resolution_pixels[2]) + " computations. To attain this accuracy with simple IRS would require " + str(n_p ** 2 ** 2) + " computations.")

print("number of final centers:", len(high_resolution_pixels[0]))