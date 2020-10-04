"""Model construction utilities for 1-D GANs"""
import tensorflow as tf

def early_gen_loss(discriminator):

    def early_gen_loss_disc(ones, generated):
        """Calculate the loss for a given set of generated data.
        Statefully depends on the discriminator model.  This is
        the log(D(G(x))) in Goodfellow's paper, except negative
        since we are trying to minimize in model.fit rather than
        maximize, as in his paper.
        """
        discriminator_assessment = discriminator(generated)
        # labels = tf.ones_like(discriminator_assessment[:,:1])
        # batch_losses = bce(labels, discriminator_assessment[:,:1])
        batch_losses = -tf.math.log(discriminator_assessment)
        return batch_losses
    return early_gen_loss_disc

def late_gen_loss(discriminator):
    def late_gen_loss_disc(ones, generated):
        """Calculate the loss for a given set of generated data.
        Statefully depends on the discriminator model.  This is
        the log(1-D(G(x))) in Goodfellow's paper.
        """
        discriminator_assessment = discriminator(generated)
        # labels = tf.zeros_like(discriminator_assessment[:,:1])
        # correct_prob = tf.ones_like(discriminator_assessment[:,:1])
        # batch_losses = bce(labels, correct_prob-discriminator_assessment[:,:1])
        batch_losses = tf.math.log(
            tf.ones_like(discriminator_assessment)-discriminator_assessment
        )
        return batch_losses
    return late_gen_loss_disc