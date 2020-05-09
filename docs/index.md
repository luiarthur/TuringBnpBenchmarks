---
layout: page
---

# This is the home page!

Check out this <a href="{{site.baseurl}}/test-page.html"> test page </a>!
It is way more interesting. 

<hr>

{% for post in site.posts %}
<div class="post-preview">
    <a href="{{ post.url | prepend: site.baseurl }}">
        <h4 class="post-title">
            {{ post.title }}
            {% if post.subtitle %}
            &mdash;
            <a class="post-subtitle">
                {{ post.subtitle }}
            </a>
            {% endif %}
        </h4>
    </a>
    <p class="post-meta">Posted on {{ post.date | date: "%-d %b, %Y" }}</p>
</div>
<hr>
{% endfor %}
