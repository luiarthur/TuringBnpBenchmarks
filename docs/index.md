---
layout: page
---

# Welcome to Turing BNP Benchmarks!

<!--
Check out this <a href="{{site.baseurl}}/test-page.html"> test page</a>!
It is way more interesting. 
-->

Hi! In this site, I benchmark several probabilistic programming languages
(PPLs) including [`Turing.jl`][1], [`STAN`][2], [`TensorFlow Probability`][3],
[`Pyro`][4], and [`NIMBLE`][5] for fitting common Bayesian nonparametric (BNP)
models. This project is funded by [Google Summer of Code][6] 2020 and led by
the `Turing` team. See the [GitHub repo][7] for more on the motivation for this
project.

My mentors for this project are [Hong Ge][8], [Martin Trapp][9], and 
[Cameron Pfiffer][10].

## Posts

{% for post in site.posts %}
<div class="post-preview">
    <a href="{{ post.url | prepend: site.baseurl }}">
        &raquo; {{ post.title }}
        {% if post.subtitle %}
        &mdash;
        <a class="post-subtitle">
            {{ post.subtitle }}
        </a>
        {% endif %}
    </a>
    <p class="post-meta" style="font-size: 16px">
       Posted on {{ post.date | date: "%-d %b, %Y" }}
    </p>
</div>
{% endfor %}


[1]: https://turing.ml/
[2]: https://mc-stan.org/
[3]: https://www.tensorflow.org/probability
[4]: https://pyro.ai/
[5]: https://r-nimble.org/
[6]: https://summerofcode.withgoogle.com/
[7]: {{site.github_repo}}
[8]: http://mlg.eng.cam.ac.uk/hong/ 
[9]: https://martintdotblog.wordpress.com/
[10]: http://cameron.pfiffer.org/
