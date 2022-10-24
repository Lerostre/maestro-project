import numpy as np
import re
from functools import reduce
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def consecutive(x):
    if len(x) == 2:
        # если два, то вырождается
        return 1 if x[-1] - x[0] == 1 else 0
    if len(x) > 2:
        return 1 if (x[0] < x[-1]) and (len(x)*(x[0]+x[-1])/2 == sum(x)) else 0

    
def search(request, sent_df, word_df, meta_df):
    sub_req = request.split()
    info = []

    el_chars = {}
    for element in sub_req:
        with_pos = False
        if element[0] == '"':
            mode = "tokens"
        elif element[0].isascii():
            mode = "poss"
        else:
            mode = "lemmas"
        if "+" in element:
            with_pos = True
        print(element, mode)
        el_chars.update({element: word_search(element=element,
                                              sent_df=sent_df,
                                              word_df=word_df,
                                              mode=mode,
                                              with_pos=with_pos)})
        
    if len(el_chars) == 1:
        sents, word_indices = list(el_chars[request].keys()), list(el_chars[request].values())
        for sent in sents:
            new_info = extract_info(sent, word_indices, meta_df, word_df, 1)
            if new_info:
                info.append(new_info)

    else:
        common_sents = list(reduce(lambda x, y: x & y.keys(), el_chars.values()))

        for sent in common_sents:
            listik = []
            for key in el_chars:
                listik.append(el_chars[key][sent])
            grid = np.array(np.meshgrid(*listik)).T.reshape(-1, len(el_chars))
            diffs = np.apply_along_axis(consecutive, arr=grid, axis=1)
            word_indices = diffs[np.where(diffs == 1)]
            if word_indices.size > 0:
                info.append(extract_info(sent, word_indices, meta_df, word_df))

    return info


def replace_keep_case(word, replacement, text):
    def func(match):
        g = match.group()
        if g.islower(): return replacement.lower()
        if g.istitle(): return replacement.title()
        if g.isupper(): return replacement.upper()
        return replacement      
    return re.sub(rf"\b{word}\b", func, text, flags=re.I)


def highlight(sentence, string):
    sent_check = sentence.lower()
    
    if not string[0].isascii() and f"<mark>{string}</mark>".lower() not in sent_check:
        return replace_keep_case(string, f'<mark>{string}</mark>', sentence)
    else:
        return sentence


def highlight2(sentence, string):
    red = "\033[31m"
    nul = "\033[0m"
    sentence = re.sub(
        r"\b({})\b".format(re.escape(string)),
        r"{}\1{}".format(red, nul),
        sentence,
        flags=re.I,
    )
    return sentence


def extract_info(sent_indice, word_indice, meta_df, word_df, num=2):
    if sent_indice:
        row = meta_df.iloc[sent_indice]
        sent = row.sentence
        if num == 1:
            words = [word_df.iloc[item].token for sublist in word_indice for item in sublist]
            for word in words:
                sent = highlight(sent, word)
        #print(word_df.iloc[word_indice].pos)
        return {"index": str(sent_indice), "sents": sent,
               "titles": row.title, "dates": row.date}
    else:
        return None
    
    
def word_search(element, sent_df, word_df, mode="tokens", with_pos=False, is_normal=False):
    
    res = {}
    
    print(element, mode, with_pos)

    if mode == "lemmas" and not is_normal:
        elements = query_parse(element)
        for el in elements:
            res.update(word_search(el, sent_df, word_df, mode, with_pos, is_normal=True))
        return res

    elif mode == "tokens":
        element = element[1:-1]

    if with_pos:
        indices = []
        word, pos = element.split("+")
        word_indices = word_search(word, sent_df, word_df, mode)
        pos_indices = word_search(pos.upper(), sent_df, word_df, mode="poss")
        indices_candidates = word_indices.keys() & pos_indices.keys()
        morph_indices = []
        for sent in indices_candidates:
            candidate = list(set(word_indices[sent]) & set(pos_indices[sent]))
            if candidate:
                indices.append(sent)
                morph_indices.append(candidate)

    else:
        v = np.vectorize(lambda x: element in x)
        indices = np.where(v(sent_df[mode]) == True)[0]
        row = sent_df.iloc[indices]
        morph_indices = row[mode].apply(lambda x: x[element])

    if not any(morph_indices):
        indices = []

    return dict(zip(indices, morph_indices))


def query_parse(word):
    variants = []

    for i in morph.parse(word):
        form = i.normal_form
        if form not in variants:
            variants.append(form)

    return variants


def morphy_converter(x):
    converter = {"ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ",
                "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB", "GRND": "VERB",
                "NUMR": "NUM", "ADVB": "ADV", "PREP": "ADP", "PTCL": "PART"}
    return converter[x] if x in converter else x