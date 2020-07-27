import re
import numpy as np
import pandas as pd

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def tosec(t):
    return round(float(re.findall(r'.*(?=s)', t)[0]))

def minsec2sec(t):
    m = re.findall(r'\d+(?=min)', t)[0]
    s = re.findall(r'\d+(?=s)', t)[0]
    return int(m) * 60 + int(s)

def ms2sec(t):
    ms = re.findall(r'.*(?=ms)', t)[0]
    return float(ms) / 1000

def sanitize_hms(t):
    x = t.split(':') 
    if len(x) == 3:
        h, m, s = x
    elif len(x) == 2:
        m, s = x
        h = 0
    else:
        NotImplemented

    return int(h) * 60 * 60 + int(m) * 60 + int(s)

def sanitize_py(t):
    if re.match(r'\d+min', t):
        return minsec2sec(t)
    elif re.match(r'\d+(\.\d+)*\s+ms', t):
        return ms2sec(t)
    else:
        return tosec(t)

def get_stan_gp_times(path):
    content = read_file(path)
    ts = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    ts = [sanitize_py(t) for t in ts]
    return dict(model='gp', ppl='stan',
                advi_compile=ts[0],
                hmc_compile=ts[0],
                nuts_compile=ts[0],
                advi_run=ts[1],
                hmc_run=ts[2],
                nuts_run=ts[3])

def get_turing_gp_times(path):
    nb_content = read_file(path)
    t = re.findall(r'\d+\.?\d+\s(?=seconds)', nb_content)
    r = re.findall(r'(?<=Time:\s)\d+:\d+:\d+', nb_content)
    t = list(map(float, t))
    r = list(map(sanitize_hms, r))
    return dict(model='dp_sb_gmm',
                ppl='turing',
                advi_compile=float(t[0] - r[0]),
                hmc_compile=float(t[1] - r[1]),
                nuts_compile=float(t[2] - r[2]),
                advi_run=r[0],
                hmc_run=r[1],
                nuts_run=r[2])

if __name__ == '__main__':
    path_to_nb="../notebooks"
    times_df = pd.DataFrame()

    # GP STAN Timings.
    stan_nb_path = '{}/gp_stan.ipynb'.format(path_to_nb)
    times = get_stan_gp_times(stan_nb_path)
    times_df = times_df.append(times, ignore_index=True)
    print(times_df)

    # Turing STAN Timings.
    turing_nb_path = '{}/gp_turing.ipynb'.format(path_to_nb)
    times = get_turing_gp_times(turing_nb_path)
    times_df = times_df.append(times, ignore_index=True)
    print(times_df)

    # Pyro STAN Timings.
    # Numpyro STAN Timings.
    # TFP STAN Timings.
    # NIMBLE STAN Timings.
