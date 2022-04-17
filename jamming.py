#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from multiprocessing import Pool, cpu_count
import typer
from functools import partial

# En honor a Ricardo Jamming

app = typer.Typer()


def msg(size):
    return np.random.binomial(1, 0.5, size=size)

def noise(size, p):
    return np.random.binomial(1, p, size=size)

def naked_experiment(size=100, p=1e-2):
    m = msg(size)
    n = noise(m.shape, p)
    return sum(m ^ n != m)

G = np.array([[1, 1, 0, 1],
              [1, 0, 1, 1],
              [1, 0, 0, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=int).T

H = np.array([[1, 0, 1, 0, 1, 0, 1],
              [0, 1, 1, 0, 0, 1, 1],
              [0, 0, 0, 1, 1, 1, 1]], dtype=int)

R = np.array([[0, 0, 0, 0],
              [0, 0, 0, 0],
              [1, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]], dtype=int).T

def coded_experiment(size=100, p=1e-2):
    # Hamming (7, 4)
    m = msg((size, 1))
    # print(f'{m.shape = }\n{m}')

    m_matrix = m.reshape(size // 4, 4)
    # print(f'{m_matrix.shape = }\n{m_matrix}')

    # Encoding
    tx_matrix = (G.T.dot(m_matrix.T) % 2).T

    tx = tx_matrix.reshape(7 * (size // 4), 1)

    # print(f'{tx.shape = }\n{tx.T}')
    # Channel
    n = noise(tx.shape, p)
    # print(n.reshape(size // 4, 7))
    rx = (tx + n) % 2
    # print(f'{rx.shape = }\n{rx.T}')

    # Decoding
    rx_matrix = rx.reshape(size // 4, 7)
    syndrome = (H.dot(rx_matrix.T) % 2).T
    corrected = rx_matrix.copy()
    # sigma = '\u03A3'
    # print(f'{"tx " + str(tx.shape):<17}\t{"noise":<17}\t{"rx":<17}\t{sigma:<3}\t{"syndrome":<10}\t{"corrected":<10}')
    for row in range(rx_matrix.shape[0]):
        tx_row = tx_matrix[row]
        rx_row = rx_matrix[row]
        # errors = sum(tx_row != rx_row)
        pos = syndrome[row, :].dot(np.array([1, 2, 4]))
        if pos > 0:
            corrected[row, pos - 1] ^= 1
        # errors_post = sum(tx_row != corrected[row, :])
        # print(f'{str(tx_row):<17}\t{str(n.reshape(25, 7)[row, :]):<17}\t{str(rx_row):<17}\t{str(errors):<3}\t{str(syndrome[row, :]):<10}\t{str(corrected[row, :]):<10}\t{str(errors_post)}')

    recv_msg = R.dot(corrected.T).T.reshape(size, 1)
    total_errors = sum((m[:, 0] - recv_msg[:, 0]) % 2)
    return total_errors


def plot_hist(rs, title, size, p):
    freqs, edges, _ = plt.hist(rs, bins=size, range=(0, size), edgecolor='black', density=True)
    plt.title(f'{title} ({len(rs)} frames de {size} bits)')
    plt.grid(True)
    plt.xlabel('# de errores en el mensaje recibido')
    plt.ylabel('Frecuencia relativa')
    plt.axis([0, binom(size, p).ppf(0.999) * 1.5, 0, 1.2])
    plt.annotate(f'Llegan bien el {100*freqs[0]:0.1f}%', (1, freqs[0]), color='red',
        bbox=dict(boxstyle="larrow,pad=0.3", fc='cyan', ec="b", lw=2))


@app.command()
def comparison(frame_size : int = 100,
               bit_error_prob : float = typer.Option(1e-2, min=0, max=1),
               sample_size : int = 10000):

    pool = Pool(cpu_count() - 1)
    naked_rs = pool.starmap(partial(naked_experiment, frame_size, bit_error_prob), [()] * sample_size)
    coded_rs = pool.starmap(partial(coded_experiment, frame_size, bit_error_prob), [()] * sample_size)

    plt.subplots(constrained_layout=True)
    plt.subplot(121)
    plot_hist(naked_rs, 'Naked',
              frame_size, bit_error_prob)
    plt.subplot(122)
    plot_hist(coded_rs, f'Hamming(7, 4) Coded (user bitrate will be {4*100/7:0.1f}%)',
              frame_size, bit_error_prob)
    plt.show()

if __name__ == '__main__':
    app()
