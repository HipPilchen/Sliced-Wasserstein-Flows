import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import trange
from matplotlib.patches import FancyArrowPatch
import cvxpy as cp
import warnings
sns.set('paper')

"""
Compute the Wasserstein distance between samples of continuous distribution

"""

myplot = lambda x,y,col: plt.scatter(x,y, s = 30, edgecolors="k", c=col, linewidths=2)

# In 2D

def Arrow2D(x, y, dx, dy, label = 'Average direction', color='k', mutation_scale=20):
    arrow = FancyArrowPatch((x, y), (x+dx, y+dy), mutation_scale=mutation_scale, color=color, label=label)
    return arrow 

def viz_2D(alpha, beta, color1 = 'b', color2 = 'r', arrow_direction = None, title = ''):
    fig, ax = plt.subplots(figsize = (10,7))
    # ax.axis("off")
    ax.scatter(alpha[0,:],alpha[1,:], s = 30, edgecolors="k",alpha = 0.5, c=color1, linewidths=2, label='Generated distribution')
    ax.scatter(beta[0,:],beta[1,:], s = 30, edgecolors="k",alpha = 0.5, c=color2, linewidths=2, label='Target distribution')
    min_lim_x = np.minimum(np.min(beta[0,:]),np.min(alpha[0,:]))
    max_lim_x = np.maximum(np.max(beta[0,:]),np.max(alpha[0,:]))
    min_lim_y = np.minimum(np.min(beta[1,:]),np.min(alpha[1,:]))
    max_lim_y = np.maximum(np.max(beta[1,:]),np.max(alpha[1,:]))
    
    if arrow_direction is not None:
        starting_arrow = np.mean(alpha, axis = 1)
        arrow_direction /= np.linalg.norm(arrow_direction)
        arrow = Arrow2D(starting_arrow[0], starting_arrow[1], 
                        arrow_direction[0], arrow_direction[1], color='k')
        ax.add_patch(arrow)
    ax.legend()
    ax.set_xlim(min_lim_x-.1,max_lim_x+.1)
    ax.set_ylim(min_lim_y-.1,max_lim_y+.1)
    if title:
        plt.suptitle(title)
    plt.show()


# In 3D 

def viz_3D(alpha, beta, color1='b', color2='r', title = '', arrow_direction = None):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_zticklabels([])
    ax.scatter(alpha[0,:], alpha[1,:], alpha[2,:], c=color1, marker='o', label='Generated distribution')

    ax.scatter(beta[0,:], beta[1,:], beta[2,:], c=color2, marker='o', label='Target distribution')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        plt.suptitle(title)
        
    if arrow_direction is not None:
        starting_arrow = np.mean(alpha, axis = 1)
        arrow_direction /= np.linalg.norm(arrow_direction)
        ax.quiver(starting_arrow[0], starting_arrow[1], starting_arrow[2], 
                  arrow_direction[0]*2, arrow_direction[1]*2, arrow_direction[2]*2, color='k', label='Average direction')

    ax.legend()
    plt.show()
    
    
"""
Compute the Wasserstein distance between samples of continuous distribution

"""

def distmat(x,y):
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def wasserstein_dist(alpha, beta):
    C = distmat(alpha,beta)
    n = alpha.shape[1]
    m = beta.shape[1]
    P = cp.Variable((n,m))
    u = np.ones((m,1))
    v = np.ones((n,1))
    U = [0 <= P, cp.matmul(P,u)==np.ones((n,1))/n, cp.matmul(P.T,v)==np.ones((m,1))/m]
    objective = cp.Minimize( cp.sum(cp.multiply(P,C)) )
    prob = cp.Problem(objective, U)
    result = prob.solve()
    warnings.filterwarnings("ignore", category=FutureWarning)
    return result

"""
Compute specific distributions

"""

def random_torus_sampler(a, c, n_samples):
    """
    c: radius from the center of the hole to the center of the torus
    a: radius of the tube
    """
    r = a * np.random.uniform(0, 1, n_samples)
    u = 2*np.pi * np.random.uniform(0, 1, n_samples)
    v = 2 * np.pi * np.random.uniform(0, 1, n_samples)
    
    return np.array([
        (c + r * np.cos(v)) * np.cos(u), # x-coordinates
        (c + r * np.cos(v)) * np.sin(u), # y-coordinates
        r * np.sin(v)])  # z-coordinates
    