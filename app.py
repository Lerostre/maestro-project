import pandas as pd
import numpy as np
import ast
from web_search import *
from flask import Flask, render_template, request, url_for, redirect
import os
from flask_paginate import Pagination

app = Flask(__name__)

meta = pd.read_csv("sents_meta.csv")
sents = pd.read_csv("parsed_sents.csv")
sents = sents.applymap(lambda x: ast.literal_eval(x))
words = pd.read_csv("morph_info.csv")

def get_users(res, offset=0, per_page=10):
    return res[offset: offset + per_page]


@app.route('/', methods=('GET','POST'))
def index():

    print(os.path.join(app.root_path, 'static'),
                               'portada.jpg')

    if request.method == 'POST':
        data = request.form['query']
        return redirect(url_for('search2', query=data))

    else:
        html = render_template('index.html')
        return html

@app.route('/search2/', defaults={'query': None}, methods=['GET', 'POST'])
@app.route('/search2/<query>', methods=['GET', 'POST'])
def search2(query):

    if request.method == 'POST':
        data = request.form['query']
        return redirect(url_for('search2', query=data))

    if not query:
        html = render_template('create.html', pagination=False, total=0)
        return html

    results = search(query, sents, words, meta)

    page, per_page, offset = get_page_args(page_parameter='page',
                                    per_page_parameter='per_page')
    per_page = 10
    total = len(results)
    pagination_users = get_users(results, offset=offset, per_page=per_page
    pagination = Pagination(page=page, per_page=per_page, total=total,
                            css_framework='bootstrap4')
    if total % 2 != 0:
        pagination_users.append([])
    items = list(zip(pagination_users[0::2], pagination_users[1::2]))
    html = render_template('create.html',
                            page=page,
                            query=query,
                            items=items,
                            per_page=10,
                            pagination=pagination,
                            total=total,
                            highlight=highlight
                            )
    return html

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
