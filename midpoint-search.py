import numpy as np 
import argparse
import json
import math
import time
from mayavi import mlab # need to install mayavi which requires qt and qmake added to path
from scipy.stats import multivariate_normal
from functools import partial
from typing import Callable, List, Tuple, Optional

# Domain
UPPER_BOUND = 2
LOWER_BOUND = -2
SPACING = 100
x = np.linspace(LOWER_BOUND, UPPER_BOUND, SPACING)
y = np.linspace(LOWER_BOUND, UPPER_BOUND, SPACING)
X, Y = np.meshgrid(x, y)
grid = np.dstack((X, Y))

# Hyperparameters
eps = 0.001 # For computing partial derivative
step_size = 0.001 # for computing next geodesic point along direction
colors = [
    (0.96,0.11,0.13), #red
    (0.96,0.3,0.11), # orange
    (1,0.99,0.42), # yellow
    (0.64,1,0.42), #yellow green
    (0.18,0.84,0.23), # green
    (0.19,0.91,0.9), # cyan
    (0.1,0.48,0.91), # blue
    (0.1,0.48,0.91), # ultramaine
    (0.35,0.17,0.91), # purple
    (0.93,0.24,0.92) #pink
]

def surface(g: Callable[[float, float], float], points: List[float]) -> float:
    '''
    Get surface points for a specified function and input of points
    '''
    if (isinstance(points[0], float)):
        return g(points[0], points[1])
    else:
        return np.asarray([surface(g, p) for p in points])

def get_surface(g: Callable[[float, float], float]) -> Callable[[float, float], float]:
    '''
    Get surface with the as a first class function to call later
    '''
    return partial(surface, g)

def get_distribution(mean: List[int], cov_matrix: List[List[int]]) -> Callable[[float, float], float]:
    '''
    Get a normal distribution with given parameters, used as the surface
    TODO(?): modify function to return different pdfs
    '''
    return multivariate_normal(mean, cov_matrix).pdf # The probability distribution as a Callable

def dfdx(f: Callable[[float, float], float], x: float, y: float) -> float:
    '''
    Approximating the partial derivative of function f w.r.t. x
    '''
    return (f([x + eps, y]) - f([x - eps, y])) / (2 * eps)

def dfdy(f: Callable[[float, float], float], x: float, y: float) -> float:
    '''
    Approximating the partial derivative of function f w.r.t. y
    '''
    return (f([x, y + eps]) - f([x, y - eps])) / (2 * eps)

def normal(f: Callable[[float, float], float], x: float, y: float) -> Tuple[int]:
    '''
    Returns the normal vector at a point
    '''
    return (-dfdx(f, x, y), -dfdy(f, x, y), 1)

def midpoint_search(
        f: Callable[[float, float], float],
        point: Tuple[float],
        direction: Tuple[float],
        distance: Optional[float] = None
    ) -> Tuple[np.array]:
    '''
    Uses the midpoint search method to compute a geodesic for
    a given surface f, the initial point, the initial direction,
    and the max distance of the geodesic to be drawn. Outputs
    the coordinates of the geodesic in 3D space.

    Paper: https://cs.stanford.edu/people/jbaek/18.821.paper2.pdf 
    '''
    x_values, y_values, f_values = None, None, None
    x0, y0 = point # starting x y
    dx0, dy0 = direction # starting direction

    eta = step_size / np.linalg.norm([dx0, dy0]) # normalization factor
    max_iter = 30000 # Upper bound on iteration
    x_values = np.zeros(max_iter)
    y_values = np.zeros(max_iter)
    f_values = np.zeros(max_iter)

    # Initial points and direction
    x_values[0], x_values[1] = x0, x0 + eta * dx0 
    y_values[0], y_values[1] = y0, y0 + eta * dy0
    f_values[0] = f([x0, y0])

    i = 1
    for i in range(1, max_iter-1):
        # Current point in space
        xt, yt = x_values[i], y_values[i]
        ft = f([xt, yt])
        f_values[i] = ft
        # Previous values
        xt_1, yt_1 = x_values[i-1], y_values[i-1]
        ft_1 = f([xt_1, yt_1])
        # Adding prior distance forward
        xs = xt + (xt - xt_1)
        ys = yt + (yt - yt_1)
        fs = ft + (ft - ft_1)
        # Surface normal's effect on change.
        df = fs - f([xs, ys]) # Measuring surface change
        n = normal(f, xt, yt) # Normal vector at xt, yt to surface
        gamma = df * n[2] # Scalar value of change from normal level curve component
        # subtracting forward by normal component change
        xtp1 = xs - gamma * n[0]
        ytp1 = ys - gamma * n[1]
        # new value update
        x_values[i+1] = xtp1
        y_values[i+1] = ytp1
        # Check for out of bounds and max distance
        if (max(xtp1, ytp1) > UPPER_BOUND or min(xtp1, ytp1) < LOWER_BOUND) or\
            (distance is not None and np.linalg.norm(np.array((xtp1, ytp1)) - np.array((x0, y0))) >= distance):
            # print("breaking", max(xtp1, ytp1), min(xtp1, ytp1))
            break
    i = min(max_iter-1, i+1)
    f_values[i] = f([x_values[i], y_values[i]])
    return x_values[:i+1], y_values[:i+1], f_values[:i+1]

def main(args):
    # Parametric Family of distribution
    try:
        with open(args.filename, 'r') as f:
            data = json.loads(f.read())
            means = data.get('means')
            covariance_matrices = data.get('covariance_matrices')
            points = data.get('points')
            directions = data.get('directions')
            distances = data.get('distances')
            dots = data.get('dots')
    except Exception as e:
        print('Error Loading Data, using default values to plot')
        print(e)
        means = [[0, 0]]
        covariance_matrices = [[[1, 0],[0, 1]]]
        points = [[(-2, -2)]]
        directions = [[(1, 1)]]
        distances = [[None]]
        dots = None
    # make sure we have the correct number of each parameter
    assert len(means) == len(covariance_matrices)
    assert len(points) == len(directions) and len(points) == len(distances)
    assert len(means) == len(points)
    # Setup plot
    print(args)
    for i in range(len(means)):
        if args.function:
            f = get_surface(lambda x, y : eval(args.function))
            z_factor = 1
        else:
            f = get_distribution(means[i], covariance_matrices[i])
            z_factor = 5
        mlab.surf(np.transpose(X), np.transpose(Y), f(grid) * z_factor, colormap='summer')
        for j, (point, direction, distance) in enumerate(zip(points[i], directions[i], distances[i])):
            # compute geodesic for surface and plot the curve
            print(f"Started: computing surface: {i+1}/{len(means)}, geodesic: {j+1}/{len(points[i])}.", end=" ")
            prev_time = time.time()
            g_x, g_y, g_f = midpoint_search(f, point, direction, distance)
            curr_time = time.time()
            print(f"Completed: time: {curr_time - prev_time:02f}")
            l = mlab.plot3d(
                g_y, g_x, g_f * z_factor,
                color=(0, 0, 0),
                # color=colors[j%len(colors)],
                line_width=1.0)
        # Plot current distribution as surface]
    if dots:
        for dot in dots:
            mlab.points3d(dot[1], dot[0], f(dot) * z_factor, color=(0,0,0), scale_factor=0.09)
    s = mlab.surf(np.transpose(X), np.transpose(Y), f(grid) * z_factor, colormap='autumn')
    mlab.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='GeodesicArt',
        description='MATH 4441 Final Art Project'
    )
    parser.add_argument('filename')
    parser.add_argument('-f', '--function')
    args = parser.parse_args()
    main(args)