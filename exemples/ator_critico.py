import collections
import gymnasium as gym
import numpy as np
import statistics
import tensorflow as tf
import tensorflow.keras.layers as layers
import tqdm

# ambiente 
env = gym.make('CartPole-v1')

# seed
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

esp = np.finfo(np.float32).eps.item()


class AtorCritico(tf.keras.Model):
    def __init__(
        self,
        num_actions: int,
        num_hidden_units: int):
        super().__init__()

        self.common = layers.Dense(num_hidden_units, activation='relu')
        self.ator = layers.Dense(num_actions)
        self.critico = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.ator(x), self.critico(x)


def env_steps(action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    estado, recompenca, final, truncado, info = env.step(action)
    return (estado.astype(np.float32),
            np.array(recompenca, np.int32),
            np.array(final, np.int32))


def tf_env_steps(action: tf.Tensor) -> list[tf.Tensor]:
    return tf.numpy_function(env_steps, [action], [tf.float32, tf.int32, tf.int32])


def rodar_ep(initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    acao_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    valores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    recompencas = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    initial_state_shape = initial_state.shape
    estado = initial_state

    for t in tf.range(max_steps):
        estado = tf.expand_dims(estado, 0)

        action_logists_t, value = model(estado)

        acao = tf.random.categorical(action_logists_t, 1)[0, 0]
        acao_probs_t = tf.nn.softmax(action_logists_t)

        valores = valores.write(t, tf.squeeze(value))

        acao_probs = acao_probs.write(t, acao_probs_t[0, acao])

        estado, recompenca, final = tf_env_steps(acao)
        estado.set_shape(initial_state_shape)

        recompencas = recompencas.write(t, recompenca)

        if tf.cast(final, tf.bool):
            break

    acoes_prob = acao_probs.stack()
    valores = valores.stack()
    recompencas = recompencas.stack()

    return acoes_prob, valores, recompencas


def receber_valor_esperado(
    recompencas: tf.Tensor,
    gamma: float,
    standardize: bool = True
) -> tf.Tensor:
    n = tf.shape(recompencas)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)

    recompencas = tf.cast(recompencas[::-1], dtype=tf.float32)
    soma_descontada = tf.constant(0.0)
    soma_descontada_shape = soma_descontada.shape
    for i in tf.range(n):
        recompenca = recompencas[i]
        soma_descontada = recompenca + gamma * soma_descontada
        soma_descontada.set_shape(soma_descontada_shape)
        returns = returns.write(i, soma_descontada)
    returns = returns.stack()[::-1]
    if standardize:
        returns = ((returns-tf.math.reduce_mean(returns)) /
        (tf.math.reduce_std(returns) + esp))
    return returns


def computar_perdas(
    action_prob: tf.Tensor,
    valores: tf.Tensor,
    returns: tf.Tensor
) -> tf.Tensor:
    vantagem = returns-valores

    action_log_prob = tf.math.log(action_prob)
    ator_loss = -tf.math.reduce_sum(action_log_prob * vantagem)

    critico_loss = huber_loss(valores, returns)

    return ator_loss+critico_loss


@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    otimizador: tf.keras.optimizers.Optimizer,
    gamma: float,
    num_max_steps_por_ep: int) -> tf.Tensor:

    with tf.GradientTape() as tape:
        acoes_prob, valores, recompencas = rodar_ep(initial_state, model,
                                                    num_max_steps_por_ep)

        retornos = receber_valor_esperado(recompencas, gamma)

        acoes_prob, valores, retornos = [
                tf.expand_dims(x, 1) for x in [acoes_prob, valores, retornos]
            ]

        perda = computar_perdas(acoes_prob, valores, retornos)
    grads = tape.gradient(perda, model.trainable_variables)

    otimizador.apply_gradients(zip(grads, model.trainable_variables))

    recompenca_do_ep = tf.math.reduce_sum(recompencas)

    return recompenca_do_ep


huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)


def main():
    otimizador = tf.keras.optimizers.Adam(learning_rate=0.01)

    num_actions = env.action_space.n
    num_hidden_units = 128

    model = AtorCritico(num_actions, num_hidden_units)

    min_ep = 100
    max_ep = 10000
    max_steps_per_ep = 500

    limiar_de_recompenca = 475
    recompenca_rodar = 0

    gamma = .99

    recompancas_do_ep: collections.deque = collections.deque(maxlen=min_ep)

    t = tqdm.trange(max_ep)

    for i in t:
        estado_inicial, info = env.reset()
        estado_inicial = tf.constant(estado_inicial, dtype=tf.float32)
        recompanca_do_ep = int(train_step(
            estado_inicial, model, otimizador, gamma, max_steps_per_ep)
        )

        recompancas_do_ep.append(recompanca_do_ep)
        recompenca_rodar = statistics.mean(recompancas_do_ep)

        t.set_postfix(recompanca_do_ep=recompanca_do_ep,
                      recompenca_rodar=recompenca_rodar)

        if recompenca_rodar > limiar_de_recompenca and i > min_ep:
            break

    print(f'\n EP:{i} \n recompanca media {recompenca_rodar:.2f}')


if __name__ == "__main__":
    main()
