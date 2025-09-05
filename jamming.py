#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from multiprocessing import Pool, cpu_count
import typer
from functools import partial
from datetime import datetime
from nominekto import summon

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

# Esta matriz se usa para CODIFICAR
# un mensaje de 4 bits
# a un CODEWORD de 7
# de los cuales 3 son de redundancia

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
    plt.title(f'{title}') # ({len(rs)} frames de {size} bits)')
    plt.grid(True)
    plt.xlabel('# de errores en el mensaje recibido')
    plt.ylabel('Frecuencia relativa')
    plt.axis([0, binom(size, p).ppf(0.999) * 1.5, 0, 1.2])
    plt.annotate(f'Llegan bien el {100*freqs[0]:0.1f}%', (1, freqs[0]), color='red',
        bbox=dict(boxstyle="larrow,pad=0.3", fc='cyan', ec="b", lw=2))

def save_fig(exp):
    now = datetime.now()
    name = summon()
    png_path = f'{name}_{exp}_{now:%Y.%m.%d.%H.%M.%S}.png'
    plt.savefig(png_path)
    typer.secho(f'Saved {png_path}.', fg=typer.colors.GREEN)


@app.command()
def comparison(frame_size : int = 100,
               bit_error_prob : float = typer.Option(1e-2, min=0, max=1),
               sample_size : int = 10000):

    pool = Pool(cpu_count() - 1)
    naked_rs = pool.starmap(partial(naked_experiment, frame_size, bit_error_prob), [()] * sample_size)
    coded_rs = pool.starmap(partial(coded_experiment, frame_size, bit_error_prob), [()] * sample_size)

    # plt.subplots(constrained_layout=False)
    plt.subplot(131)
    plot_hist(naked_rs, 'Sin Código',
              frame_size, bit_error_prob)
    plt.subplot(133)
    plot_hist(coded_rs, f'Hamming(7, 4)', # Coded (user bitrate will be {4*100/7:0.1f}%)',
              frame_size, bit_error_prob)

    save_fig('comparison')

@app.command()
def probs(frame_size: int = 100,
          sample_size: int = 10000):

    # beps = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.2]
    beps = np.logspace(-4, -1.5, 50)
    rs = {}
    with typer.progressbar(beps) as prog:
        for bep in prog:
            #typer.secho(f'Experimento para prob de error de bit {bep}.')
            pool = Pool(cpu_count() - 1)
            naked_rs = pool.starmap(partial(naked_experiment, frame_size, bep), [()] * sample_size)
            coded_rs = pool.starmap(partial(coded_experiment, frame_size, bep), [()] * sample_size)
            rs[bep] = (100 * sum(x == 0 for x in naked_rs)/sample_size, 100 * sum(x == 0 for x in coded_rs)/sample_size)

    plt.plot(beps, [rs[b][0] for b in beps], '.-', label='sin')
    plt.plot(beps, [rs[b][1] for b in beps], '.-', label='con')
    plt.title('Comparación código Hamming 7,4')
    plt.xlabel('Probabilidad de Error de bit')
    plt.ylabel('% de mensajes sin error')
    plt.text(0, 25, f'{frame_size = }')
    plt.text(0, 15, f'{sample_size = }')
    plt.grid()
    plt.legend()
    save_fig('probs')

if __name__ == '__main__':
    app()
