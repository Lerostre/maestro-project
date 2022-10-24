import numpy as np
import re
from functools import reduce
import pymorphy2

morph = pymorphy2.MorphAnalyzer()


def word_search(element, sent_df, word_df, mode="tokens", with_pos=False, is_normal=False):
    """
    Базовая функция, которая ищет одно слово, в зависимости от типа запроса.
    Тип определяется правилами, указанными на странице поиска, там всё прямолинейно

    element: одно слово из запроса, которое надо найти
    sent_df: БД с полным составом предложения - леммы, слова, части речи
    word_df: БД с морфоразбором каждого слова
    mode: где искать: в леммах, словоформах, частях речи
    with_pos: обрабатывает случаи элемент+ЧР
    is_normal: если ищем по лемме, то нужно обработать все варианты
               морфологического разбора, параметр указывает, надо ли

    return: массив предложения-вхождения слова
    """

    if mode == "lemmas" and not is_normal:
        # для каждого возможного разбора надо найти все вхождения
        res = {}
        elements = query_parse(element)
        for el in elements:
            res.update(word_search(el, sent_df, word_df, mode, with_pos, is_normal=True))
        return res

    elif mode == "tokens" and not with_pos:
        element = element[1:-1]

    if with_pos:
        # если в запросе уточняется ЧР, надо её найти и замёрджить
        # результат поиска для ЧР и элемента перед плюсом
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
        # основная часть - проверяется, что элемент есть в предложении,
        # как set'e, потом достаётся, что это именно за слова, в структуре датафрейма
        # это просто ключи искомых элементов
        v = np.vectorize(lambda x: element in x)
        indices = np.where(v(sent_df[mode]) == True)[0]
        row = sent_df.iloc[indices]
        morph_indices = row[mode].apply(lambda x: x[element])

    if not any(morph_indices):
        # если ничего не найдено, ничего возвращать не надо
        indices = []

    return dict(zip(indices, morph_indices))


def query_parse(word):
    # довесок к предыдущей функции, возвращает все разборы, если
    # поиск по лемме

    variants = []
    for i in morph.parse(word):
        form = i.normal_form
        if form not in variants:
            variants.append(form)

    return variants


def morphy_converter(x):
    # для обработки запроса пользователя используется pymorphy, у него другой тегсет
    converter = {"ADJF": "ADJ", "ADJS": "ADJ", "COMP": "ADJ",
                "INFN": "VERB", "PRTF": "VERB", "PRTS": "VERB", "GRND": "VERB",
                "NUMR": "NUM", "ADVB": "ADV", "PREP": "ADP", "PTCL": "PART"}
    return converter[x] if x in converter else x


def is_ordered(nums):
    # проверяет, что массив строго возрастает
    if np.array_equal(np.sort(nums), nums):
        return True
    else:
        return False


def consecutive(l):
    # проверяет, что массив из последовательных чисел
    setl = set(l)
    M, m = max(l), min(l)
    return len(l) == len(setl) and setl == set(range(m, M+1)) and is_ordered(l)


def search(request, sent_df, word_df, meta_df):
    """
    Функция поиска по всей БД
    request: запрос, введённый в поле поиска
    sent_df: БД с полным составом предложения - леммы, слова, части речи
    word_df: БД с морфоразбором каждого слова
    meta_df: БД с полной внешней инфой по предложению

    return info: индекс + метаинформация предложения с реквестом
    """
    request = request.strip(" ?!.,''#$@%^()*&")  # защита от дурака
    sub_req = request.split()
    info = []

    el_chars = {}
    # сначала выполняется поиск по одиночнмоу элементу
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
        el_chars.update({element: word_search(element=element,
                                              sent_df=sent_df,
                                              word_df=word_df,
                                              mode=mode,
                                              with_pos=with_pos)})

    # если элемент один, то можно выводить инфу сразу
    if len(el_chars) == 1:
        sents, word_indices = list(el_chars[request].keys()), list(el_chars[request].values())
        for sent in sents:
            new_info = extract_info(sent, meta_df)
            if new_info:
                info.append(new_info)

    # если нет, то сперва надо проверить, какие предложения у элементов общие
    else:
        common_sents = list(reduce(lambda x, y: x & y.keys(), el_chars.values()))
        # теперь нужно проверить, что в этих предложениях всё действительно стоит рядом
        for sent in common_sents:
            temp = []  # собираем индексы найденных слов
            for key in el_chars:
                temp.append(el_chars[key][sent])
            # получаю все кортежи слов
            grid = np.array(np.meshgrid(*temp)).T.reshape(-1, len(el_chars))
            # проверяю каждый кортеж на то, что он последователен
            diffs = np.apply_along_axis(consecutive, arr=grid, axis=1)
            word_indices = diffs[np.where(diffs == 1)]
            # если это подтвердилось, достаём метаинфу предложения
            if word_indices.size > 0:
                info.append(extract_info(sent, meta_df))

    # выводим всё, что нашли
    return info


def extract_info(sent_indice, meta_df):
    # залезает и достаёт метаинфу для определённого предложения
    if sent_indice:
        row = meta_df.iloc[sent_indice]
        return {"index": str(sent_indice), "sents": row.sentence,
               "titles": row.title, "dates": row.date}
    else:
        return None


def highlight(sentence, string):
    # выделяет всё, что было найдено функцией search
    sent_check = sentence.lower()

    if not string[0].isascii() and f"<mark>{string}</mark>".lower() not in sent_check:
        return replace_keep_case(string, f'<mark>{string}</mark>', sentence)
    else:
        return sentence


def replace_keep_case(word, replacement, text):
    # нужно, чтобы выделение сохраняло регистр
    def func(match):
        g = match.group()
        if g.islower(): return replacement.lower()
        if g.istitle(): return replacement.title()
        if g.isupper(): return replacement.upper()
        return replacement
    return re.sub(rf"\b{word}\b", func, text, flags=re.I)