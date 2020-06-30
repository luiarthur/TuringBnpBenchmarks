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

def sanitize_py(t):
    if re.match(r'\d+min', t):
        return minsec2sec(t)
    elif re.match(r'\d+(\.\d+)*\s+ms', t):
        return ms2sec(t)
    else:
        return tosec(t)

def get_dp_sb_gmm_stan_times(path):
    nb_content = read_file(path)
    t = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    return dict(model='dp_sb_gmm',
                ppl='stan',
                advi_compile=sanitize_py(t[0]),
                hmc_compile=sanitize_py(t[0]),
                nuts_compile=sanitize_py(t[0]),
                advi_run=sanitize_py(t[1]),
                hmc_run=sanitize_py(t[2]),
                nuts_run=sanitize_py(t[3]))

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

def get_dp_sb_gmm_turing_times(path):
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

def get_dp_sb_gmm_tfp_times(path):
    nb_content = read_file(path)
    t = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    t = list(map(sanitize_py, t))
    return dict(model='dp_sb_gmm',
                ppl='tfp',
                advi_compile=np.nan,
                hmc_compile=0,
                nuts_compile=0,
                advi_run=np.nan,
                hmc_run=t[1],
                nuts_run=t[2])

def get_dp_sb_gmm_pyro_times(path):
    nb_content = read_file(path)
    t = re.findall(r'(?<=\[)\d+[:\d+]+', nb_content)
    t = list(map(sanitize_hms, t))
    return dict(model='dp_sb_gmm',
                ppl='pyro',
                advi_compile=0,
                hmc_compile=0,
                nuts_compile=0,
                advi_run=t[0],
                hmc_run=t[1],
                nuts_run=t[2])

def get_dp_sb_gmm_numpyro_times(path):
    nb_content = read_file(path)
    r = re.findall(r'(?<=\[)\d+[:\d+]+', nb_content)
    t = re.findall(r'(?<=Wall time:\s).*(?=\\n)', nb_content)
    r = list(map(sanitize_hms, r))
    t = list(map(sanitize_py, t))
    return dict(model='dp_sb_gmm',
                ppl='numpyro',
                advi_compile=np.nan,
                hmc_compile=t[0] - r[0],
                nuts_compile=t[1] - r[1],
                advi_run=np.nan,
                hmc_run=r[0],
                nuts_run=r[1])



if __name__ == '__main__':
    path_to_nb="../notebooks"
    times_df = pd.DataFrame()

    # DP SB GMM STAN TIMES
    path_to_dp_sb_gmm_stan_nb = '{}/dp_sb_gmm_stan.ipynb'.format(path_to_nb)
    dp_sb_gmm_stan_times = get_dp_sb_gmm_stan_times(path_to_dp_sb_gmm_stan_nb)
    times_df = times_df.append(dp_sb_gmm_stan_times, ignore_index=True)

    # DP SB GMM Turing TIMES
    path_to_dp_sb_gmm_turing_nb = '{}/dp_sb_gmm_turing.ipynb'.format(path_to_nb)
    dp_sb_gmm_turing_times = get_dp_sb_gmm_turing_times(path_to_dp_sb_gmm_turing_nb)
    times_df = times_df.append(dp_sb_gmm_turing_times, ignore_index=True)

    # DP SB GMM TFP TIMES
    path_to_dp_sb_gmm_tfp_nb = '{}/dp_sb_gmm_tfp.ipynb'.format(path_to_nb)
    dp_sb_gmm_tfp_times = get_dp_sb_gmm_tfp_times(path_to_dp_sb_gmm_tfp_nb)
    times_df = times_df.append(dp_sb_gmm_tfp_times, ignore_index=True)

    # DP SB GMM PYRO TIMES
    path = '{}/dp_sb_gmm_pyro.ipynb'.format(path_to_nb)
    times = get_dp_sb_gmm_pyro_times(path)
    times_df = times_df.append(times, ignore_index=True)

    # DP SB GMM NUMPYRO TIMES
    path = '{}/dp_sb_gmm_numpyro.ipynb'.format(path_to_nb)
    times = get_dp_sb_gmm_numpyro_times(path)
    times_df = times_df.append(times, ignore_index=True)

    # DP CRP GMM NIMBLE TIMES
    # DP CRP GMM TURING TIMES
    
    ppl = times_df.pop("ppl")
    model = times_df.pop("model")

    times_df.insert(0, "model", model)
    times_df.insert(0, "ppl", ppl)
    times_df.round().to_csv('../timings/timings.csv', index=False, na_rep='NA')

