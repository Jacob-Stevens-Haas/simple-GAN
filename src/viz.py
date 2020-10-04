"""Simple utilities for visualizing 1-D GANs
"""
from typing import List

import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

from tensorflow.python.keras.engine.training import Model

def distro_plot(real_samples: np.ndarray, generator: Model,
    discriminator: Model):
    """Plot a histogram of discriminator's ability

    Args:
        real_samples (np.ndarray): A batch-size x 1 array of real samples
        generator (Model): The generator model.  Must take a 1-D input
        discriminator (Model): The discriminator model.  Must take a 1-D input
    """
    inputs = np.random.random((real_samples.shape[0],1))
    fake_samples = generator(inputs)
    groups = np.hstack((real_samples, fake_samples.numpy()))
    plt.hist(
        groups,
        density=True,
        color=['blue', 'red'], 
        label=['True', 'Generated'],
        bins=20,
        rwidth=1
    )
    plt.title('Desitribution of True and Generated Samples')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()

def discriminator_distro_plot(real_samples: np.ndarray, generator: Model,
    discriminator: Model):
    """Plot a histogram of discriminator's ability

    Args:
        real_samples (np.ndarray): A batch-size x 1 array of real samples
        generator (Model): The generator model.  Must take a 1-D input
        discriminator (Model): The discriminator model.  Must take a 1-D input
    """
    inputs = np.random.random((real_samples.shape[0],1))
    fake_samples = generator(inputs)
    fake_disc = discriminator(fake_samples).numpy()
    real_disc = discriminator(real_samples).numpy()
    groups = np.hstack((real_disc, fake_disc))
    plt.hist(groups,
        density=True,
        color=['blue', 'red'],
        label=['True', 'Generated'],
        rwidth=1,
    )
    plt.title('Discriminator Performance')
    plt.xlabel('Assessed Probability of Being Genuine')
    plt.ylabel('Probability of Discriminator Output')
    plt.legend()

def discriminator_shape_plot(real_samples: np.ndarray, generator: Model,
    discriminator: Model, plot_range=None):
    """Plot a the discriminator function

    Args:
        real_samples (np.ndarray): A batch-size x 1 array of real samples
        generator (Model): The generator model.  Must take a 1-D input
        discriminator (Model): The discriminator model.  Must take a 1-D input
        plot_range (tuple) : The min and max to override automatic plotting
    """
    inputs = np.random.random((real_samples.shape[0],1))
    fake_samples = generator(inputs)
    # if plot_range is not None:
    minimum = min(min(real_samples), min(fake_samples.numpy()))
    maximum = max(max(real_samples), max(fake_samples.numpy()))
    # else:
    #     minimum = plot_range(0)
    #     maximum = plot_range(1)
    x = np.linspace(minimum,maximum,100)
    y = discriminator(
        tf.constant(x.reshape(-1,1))).numpy().reshape((-1))
    plt.plot(x,y, label = 'Discriminator', color='blue')
    plt.title('Discriminator Function')
    plt.xlabel('Value')
    plt.ylabel('Assessed Probability of Being Genuine')
    plt.legend()

def generator_cdf(real_samples: np.ndarray, generator: Model,
    discriminator: Model):
    """Compare the CDF of real data vs the generated data

    Args:
        real_samples (np.ndarray): A batch-size x 1 array of real samples
        generator (Model): The generator model.  Must take a 1-D input
        discriminator (Model): The discriminator model.  Must take a 1-D input
    """
    inputs = np.random.random((real_samples.shape[0],1))
    fake_samples = generator(inputs)
    real_sorted = np.sort(real_samples.flatten())
    real_cdf = (np.ones(real_samples.shape[0])/real_samples.shape[0]).cumsum()
    fake_sorted = np.sort(fake_samples.numpy().flatten())
    fake_cdf = (np.ones(fake_sorted.shape[0])/fake_sorted.shape[0]).cumsum()
    plt.plot(real_sorted, real_cdf, color='blue', label='True')
    plt.plot(fake_sorted, fake_cdf, color='red', label='Generated')
    plt.title('Generator Performance: Cumulative Distribution Function')
    plt.xlabel('Height in Inches')
    plt.ylabel('Percent shorter than x')
    plt.legend()

def quad_plot_GAN(real_samples: np.ndarray, generator: Model,
    discriminator: Model, figsize: List[int]=[8,8]):
    """A quad plot to assess GAN performance

    Args:
        real_samples (np.ndarray): A batch-size x 1 array of real samples
        generator (Model): The generator model.  Must take a 1-D input
        discriminator (Model): The discriminator model.  Must take a 1-D input
        figsize (List)
    """
    fig = figure = plt.figure(figsize=figsize)
    fig.suptitle('Snapshot of GAN performance', size = 'x-large', weight='heavy')
    plt.subplot(2,2,1)
    distro_plot(real_samples, generator, discriminator)
    plt.subplot(2,2,2)
    discriminator_distro_plot(real_samples, generator, discriminator)
    plt.subplot(2,2,3)
    discriminator_shape_plot(real_samples, generator, discriminator)
    plt.subplot(2,2,4)
    generator_cdf(real_samples, generator, discriminator)
    plt.tight_layout()