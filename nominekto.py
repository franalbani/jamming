from random import choice


ADJETIVOS = [ 'attenuated', 'noisy', 'amplifyed', 'oversampled', 'undersampled',
'accelerated', 'downsampled', 'upconverted', 'downconverted', 'modulated',
'demodulated', 'shifted', 'sliced', 'stopped', 'killed', 'paused', 'bypassed',
'halted', 'slow', 'fast', 'aliased', 'filtered', 'unfiltered', 'sharp', 'broad',
'narrow', 'recursive', 'hardcoded' ]


APELLIDOS = [
        # Mundo antiguo
        'pythagoras', # por intuir que el mundo está hecho de armonías
        'euclid',     # por formalizar la geometría
        'apollonius', # por los epiciclos, precursores de fourier
        'archimedes', # por ser precursor del análisis matemático

        # Siglo XVII
        'newton', # por la universalidad de las leyes físicas, la óptica y el análisis matemático

        # Siglo XVIII
        'euler', # por la relación entre exponenciales complejas y senoidales
        'gauss', # por anticiparse a la FFT

        # Siglo XIX
        'fourier', # por la descomposición en senoidales de señales arbitrarias
        'poisson', # por avances en análisis de Fourier
        'lovelace', # por intuir la universalidad de las computadoras
        'hertz',
        'faraday', # por la inducción magnética
        'ampere',
        'marconi',
        'morse', # por el código
        'helmholtz', # por la formulación moderna de las ecuaciones de Maxwell
        'maxwell', # por la unificación del electromagnetismo y descubrir las ondas EM
        'volta', # por las pilas
        'ohm', # por sus avances en meditación
        'hilbert', # por la transformada que extre amplitu y fase

        # Siglo XX pre 2da Guerra Mundial
        'tesla',
        'lamarr', # por spread spectrum
        'turing', # por la universalidad de la computación
        'hartley', #
        'gabor', # por los átomos de información
        'whittaker', # por la fórmula de interpolación
        'shannon', # por la teoría de la información y el teorema de muestreo
        'nyquist', # por ser precursor del teorema de muestreo
        'wiener', # por los procesos estocásticos
        'khinchin', # por el teorema sobre densidad espectral
        'hamming', # por los códigos
        'dirac', # por sus deltas

        # Siglo XX Post 2da Guerra Mundial
        'thévenin',
        'clarke', # por proponer satélites geo estacionarios
        'cooley', # por la FFT
        'tukey', # por la FFT y la palabra bit
        'harris', # por la técnicas multirate
        'viterbi', # por el algoritmo de soft decoding
        'reed', # por los códigos
        'solomon', # por los códigos
        'friss', # por la ecuación de transmisión
        'lamport', # por traer la relatividad a los sistemas distribuídos
        'torvalds', # por Linux
        'mitola', # por las SDR
        'ettus', # por los USRP
        ]


def q():
    return len(APELLIDOS) * len(ADJETIVOS)


def summon():
    return choice(ADJETIVOS) + '_' + choice(APELLIDOS)
