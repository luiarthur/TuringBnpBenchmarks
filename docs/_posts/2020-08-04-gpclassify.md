---
layout: page
title: "Gaussian Process Classification Model in various PPLs"
subtitle: GPC in PPLs
math: on
nburl: "https://github.com/luiarthur/TuringBnpBenchmarks/blob/master/src/gp-classify/notebooks/"
ppls: "Turing,STAN,TFP,Pyro,Numpyro"
sorttable: on
date_last_modified: "21 August, 2020."
---

<!--
Tables were generated via:
https://jekyllrb.com/tutorials/csv-to-table/
-->


# {{page.title}}

Last updated: {{ page.date_last_modified }}

***

<img src="{{ "/assets/img/gp-classify/data.png" | prepend: site.baseurl }}"
     class="center" alt="data image"/>

<img src="{{ "/assets/img/gp-classify/uq.png" | prepend: site.baseurl }}"
     class="center" alt="UQ for GP classification"/>

<img src="{{ "/assets/img/gp-classify/kernel_params.png" |
             prepend: site.baseurl }}"
     class="center" alt="kerenl parameters posterior"/>

## PPL Comparisons
<!-- Buttons Div for appending buttons-->
<div id="ppl-buttons" class="btn-group" role="group" aria-label="...">
</div>

{% assign ppl_array = page.ppls | split: ',' %}

{% for ppl in ppl_array %}
  {% assign ppl_lower = ppl | downcase %}

<div class="ppl-code hide" id="{{ppl_lower}}">
<p>
  <a href="{{page.nburl}}/gp_classify_{{ppl_lower}}.ipynb">Full {{ppl}} Example (notebook)</a>
</p>
    {% if ppl_lower == "turing" %}
      {% highlight julia linenos %}{% include_relative ppl-snippets/gp-classify/gp_classify_{{ppl_lower}}.jl %}{% endhighlight %}
    {% else  %}
      {% highlight python linenos %}{% include_relative ppl-snippets/gp-classify/gp_classify_{{ppl_lower}}.py %}{% endhighlight %}
    {% endif %}
</div>
{% endfor %}

#

## Timings


<table class="table table-bordered table-hover table-condensed sortable"
       id="gp-ppl-times">
  {% for row in site.data.timings.gpclassify.gpclassify_ppl_timings %}
    {% if forloop.first %}
    <tr>
      {% for pair in row %}
        <th>{{ pair[0] }}</th>
      {% endfor %}
    </tr>
    {% endif %}

    {% tablerow pair in row %}
      {{ pair[1] }}
    {% endtablerow %}
  {% endfor %}
</table>

<!-- Scripts code chunk buttons -->
<script>
$(document).ready(function(){
  // PPLs to benchmark.
  var ppls = ['Turing', 'STAN', 'TFP', 'Pyro', 'Numpyro'];

  for (ppl of ppls) {
    let ppl_lower = ppl.toLowerCase();

    // Create buttons.
    $('#ppl-buttons').append(`
      <button type="button" class="btn btn-default ${ppl_lower}">${ppl}</button>
    `);

    // Show Turing example by default.
    $("#turing").attr("class", `ppl-code show`);

    // Button callbacks. 
    $(`button.${ppl_lower}`).click(() => {
      $(".ppl-code").attr("class", `ppl-code hide`);
      $(`#${ppl_lower}`).attr("class", `ppl-code show`);
    });
  }
});
</script>


[1]: http://www.gaussianprocess.org/gpml/
[2]: http://www.gaussianprocess.org/gpml/chapters/RW2.pdf
[3]: https://aws.amazon.com/ec2/instance-types/c5/
