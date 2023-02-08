import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


def create_initializer(seed=None):
    return tf.keras.initializers.HeNormal(seed=seed)


class func_T(tf.keras.Model):
    def __init__(self, hidden_size=100):
        super(func_T, self).__init__()
        self.hidden_size = hidden_size

    def compile(self, opt):
        super(func_T, self).compile()
        self.opt = opt

    def build(self, input_shape):
        self.Layers = [layers.Dense(units=self.hidden_size,
                                    kernel_initializer=create_initializer(),
                                    activation='linear'),
                       layers.Activation(tf.nn.elu),

                       layers.Dense(units=self.hidden_size,
                                    kernel_initializer=create_initializer(),
                                    activation="linear"),
                       layers.Activation(tf.nn.elu),

                       layers.Dense(units=1,
                                    kernel_initializer=create_initializer(),
                                    activation='linear')
                       ]

    @tf.function
    def call(self, inputs, training=False):
        x = inputs
        for layer in self.Layers:
            x = layer(x, training=training)
        return x

    @tf.function
    def MINE(self, inputs, ema_eT, rate=0.001):
        joint, marginal = inputs
        with tf.GradientTape() as tape:
            joint_T = self(joint, training=True)
            marginal_T = self(marginal, training=True)

            joint_T = tf.reduce_mean(joint_T)
            exp_T = tf.reduce_mean(tf.math.exp(marginal_T))

            lower_bound = joint_T - tf.math.log(exp_T)

            ema_eT = (1 - rate) * ema_eT + rate * exp_T
            loss = -(joint_T - (exp_T / tf.stop_gradient(ema_eT)))
            grad = tape.gradient(loss, self.Layers.trainable_weights)
            self.opt.apply_gradients(zip(grad, self.Layers.trainable_weights))
            return loss, lower_bound, ema_eT

if __name__ == "__main__":
    # Toy Example
    X = np.random.uniform(low=-1, high=1, size=(300, 1))

    dice = np.arange(1, 7)
    Z = np.random.choice(dice, (300, 1)).astype(np.float64)

    joint = np.concatenate([X, Z], axis=-1)
    joint = tf.data.Dataset.from_tensor_slices(joint).batch(300)
    marginal = joint.map(lambda X: tf.concat([X[:, 0][:, tf.newaxis], tf.random.shuffle(X[:, 1])[:, tf.newaxis]], axis=1))

    inputs = tf.data.Dataset.zip((joint, marginal))
    T = func_T(hidden_size=400)
    T.compile(opt=tf.keras.optimizers.Adam(1e-4))
    iter_num = 10000
    ema_eT = tf.constant(0.0)
    losses = []
    lb = []
    ema = []
    for i in range(iter_num):
        print(f"Iteration {i + 1}:")
        for count, batch in enumerate(inputs):
            loss, lower_bound, ema_eT = T.MINE(batch, ema_eT=ema_eT)
            losses.append(loss.numpy())
            lb.append(lower_bound.numpy())
            ema.append(ema_eT.numpy())
            print("  Lower Bound:",lower_bound.numpy())
            print("  Loss       :",loss.numpy())
            print("  EMA        :", ema_eT.numpy(), "\n")

