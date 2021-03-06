import re
import numpy as np
import pandas as pd

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content

def tosec(t):
    return round(float(re.findall(r'.*(?=s)', t)[0]), 3)

def minsec2sec(t):
    m = re.findall(r'\d+(?=min)', t)[0]
    _s = re.findall(r'\d+(?=s)', t)
    if len(_s) > 0:
        s = _s[0]
    else:
        s = 0
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
    nb_content = read_file(path)
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
    # r = re.findall(r'(?<=Time:\s)\d+:\d+:\d+', nb_content)
    t = list(map(float, t))
    # r = list(map(sanitize_hms, r))
    return dict(model='gp',
                ppl='turing',
                advi_compile=t[0],
                hmc_compile=t[2],
                nuts_compile=t[4],
                advi_run=t[1],
                hmc_run=t[3],
                nuts_run=t[5])

def get_pyro_gp_times(path):
    nb_content = read_file(path)
    t = re.findall(r'(?<=\[)\d+[:\d+]+', nb_content)
    t = list(map(sanitize_hms, t))
    return dict(model='gp',
                ppl='pyro',
                advi_compile=0,
                hmc_compile=0,
                nuts_compile=0,
                advi_run=t[6],
                hmc_run=t[1],
                nuts_run=t[4])

def get_numpyro_gp_times(path):
    nb_content = read_file(path)
    r = re.findall(r'(?<=\[)\d+[:\d+]+', nb_content)
    t = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    r = list(map(sanitize_hms, r))
    t = list(map(sanitize_py, t))
    return dict(model='gp',
                ppl='numpyro',
                advi_compile=t[2],
                hmc_compile=t[0] - r[0],
                nuts_compile=t[1] - r[2],
                advi_run=t[3],
                hmc_run=r[0],
                nuts_run=r[2])

def get_tfp_gp_times(path):
    nb_content = read_file(path)
    t = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    print(t)
    t = list(map(sanitize_py, t))
    return dict(model='gp',
                ppl='tfp',
                advi_compile=0,
                hmc_compile=t[0],
                nuts_compile=t[2],
                advi_run=t[4],
                hmc_run=t[1],
                nuts_run=t[3])


if __name__ == '__main__':
    path_to_nb="../notebooks/gp_classify"
    times_df = pd.DataFrame()

    # STAN GP Timings.
    stan_nb_path = f'{path_to_nb}_stan.ipynb'
    times = get_stan_gp_times(stan_nb_path)
    times_df = times_df.append(times, ignore_index=True)

    # Turing GP Timings.
    turing_nb_path = f'{path_to_nb}_turing.ipynb'
    times = get_turing_gp_times(turing_nb_path)
    times_df = times_df.append(times, ignore_index=True)

    # Pyro GP Timings.
    pyro_nb_path = f'{path_to_nb}_pyro.ipynb'
    times = get_pyro_gp_times(pyro_nb_path)
    times_df = times_df.append(times, ignore_index=True)

    # Numpyro GP Timings.
    numpyro_nb_path = f'{path_to_nb}_numpyro.ipynb'
    times = get_numpyro_gp_times(numpyro_nb_path)
    times_df = times_df.append(times, ignore_index=True)

    # TFP GP Timings.
    tfp_nb_path = f'{path_to_nb}_tfp.ipynb'
    times = get_tfp_gp_times(tfp_nb_path)
    times_df = times_df.append(times, ignore_index=True)

    # TODO: NIMBLE GP Timings.

    # Save CSV
    times_df = times_df[['ppl',
                         'advi_run', 'hmc_run', 'nuts_run',
                         'advi_compile', 'hmc_compile', 'nuts_compile']]
    times_df = times_df.rename(columns={"ppl": "PPL",
                                        "advi_run": "ADVI (run)",
                                        "hmc_run": "HMC (run)",
                                        "nuts_run": "NUTS (run)",
                                        "advi_compile": "ADVI (compile)",
                                        "hmc_compile": "HMC (compile)",
                                        "nuts_compile": "NUTS (compile)"})
    times_df = times_df.round(3)
    times_df.to_csv('../timings/timings.csv', index=False, na_rep='NA')

    print(times_df)
    print('Written to ../timings/timings.csv')
