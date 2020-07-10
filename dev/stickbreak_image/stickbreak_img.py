import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6)
n = 5
a = 3
lw = 10

stick_length = 1

fig, ax = plt.subplots()

fs = 12

ax.plot([0, stick_length], [n + 1, n + 1], lw=lw, color='black')
ax.text(0.45, n + 1.5, r'unit stick', fontsize=fs)

for i in range(n, 0, -1):
    k = n - i + 1
    v = np.random.beta(1, a)
    lower = stick_length * (1 - v)
    upper = stick_length
    ax.plot([lower, upper], [i, i], lw=lw, color='red')

    ax.text((lower + upper) / 2, i - 0.5, r'$w_{}$'.format(k), fontsize=fs)

    ax.plot([0, stick_length * (1 - v)], [i, i], lw=lw, color='black')
    stick_length *= (1 - v)
    # if k > 1:
    #     ax.text(stick_length * 0.2, i - 0.5, r'$(1 - w_{})\cdot v_{}$'
    #             .format(k - 1, k), fontsize=fs)

ax.text(stick_length * 0.3, -0.5, r"$\vdots$", fontsize=30, color='grey')
fig.patch.set_visible(False)
ax.axis('off')
plt.ylim(0)
plt.savefig("../../docs/assets/img/dp-sb-gmm/sb.png")
plt.close()

