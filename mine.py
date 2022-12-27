import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers


def create_initializer(seed=None):
    # return tf.keras.initializers.GlorotUniform(seed=seed)
    return tf.keras.initializers.HeNormal(seed=seed)
    # return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)

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
                       # layers.Activation(tf.nn.leaky_relu),
                       # layers.Activation(tf.nn.relu),

                       layers.Dense(units=self.hidden_size,
                                    kernel_initializer=create_initializer(),
                                    activation="linear"),
                       layers.Activation(tf.nn.elu),
                       # layers.Activation(tf.nn.leaky_relu),
                       # layers.Activation(tf.nn.relu),

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
    # Test
    T = func_T(hidden_size=400)
    T.compile(opt=tf.keras.optimizers.Adam(1e-4))
    X = np.zeros((300, 2))
    X[:, 0] = np.random.uniform(low=-1, high=1, size=300)
    eps = np.random.normal(loc=0.0, scale=1.0, size=300)
    rho = 0.0

    # X[:, 0] = X[:, 0] ** 3
    # X[:, 0] = np.sin(X[:, 0])
    X[:, 1] = X[:, 0] + rho * eps

    # X = np.random.multivariate_normal(mean=[0.0, 0.0], cov=[[1.0, 0.0],
    #                                                         [0.0, 1.0]],
    #                                   size=300)

    # dice = np.arange(1, 7)
    # X[:, 1] = np.repeat(dice, 50)
    # np.random.shuffle(X[:, 1])

    joint = tf.data.Dataset.from_tensor_slices(X).batch(300)

    # index = np.random.choice(300, size=300, replace=False)
    # marginal = np.concatenate([X[:, 0].reshape(-1, 1), X[index, 1].reshape(-1, 1)], axis=1)
    # marginal = tf.data.Dataset.from_tensor_slices(marginal)

    X1 = tf.data.Dataset.from_tensor_slices(X[:, 0]).batch(300)
    X2 = tf.data.Dataset.from_tensor_slices(X[:, 1]).shuffle(512).batch(300)
    marginal = tf.data.Dataset.zip((X1, X2))
    marginal = marginal.map(lambda x1, x2: tf.transpose(tf.concat([[x1, x2]], axis=0)))  # one dimension

    inputs = tf.data.Dataset.zip((joint, marginal))
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


    def MA(lb, window_size=200):
        return [np.mean(lb[i : i + window_size]) for i in range(0, (len(lb) - window_size) + 1)]

    ma_lb = MA(lb)

    plt.style.use("bmh")
    plt.figure(1)
    plt.title("Loss")
    plt.plot(range(iter_num), losses)

    plt.figure(2)
    plt.title("Lower Bound")
    plt.plot(range(iter_num), lb)

    plt.figure(3)
    plt.title("MA Lower Bound")
    plt.xlim(0, iter_num - 1)
    plt.plot(range(len(ma_lb)), ma_lb)

    plt.figure(4)
    plt.title("EMA")
    plt.plot(range(iter_num), ema)

    plt.show()
