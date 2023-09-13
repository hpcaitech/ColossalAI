import re

def detect_lang_naive(s):
    remove_nota = u'[’·°–!"#$%&\'()*+,-./:;<=>?@，。?★、…【】（）《》？“”‘’！[\\]^_`{|}~]+'
    s = re.sub(remove_nota, '', s)
    s = re.sub('[0-9]', '', s).strip()
    res = re.sub('[a-zA-Z]', '', s).strip()
    if len(res)<=0:
        return 'en'
    else:
        return 'zh'
