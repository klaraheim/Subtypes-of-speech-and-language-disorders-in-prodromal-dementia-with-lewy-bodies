import os
import re
import whisper
import parselmouth
from statistics import mean
import stanza
from collections import Counter
import math

# Lexikalni parametry ########################################################
def tokenize(text: str):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def analyze_text(text: str):
    words = tokenize(text)
    total_tokens = len(words)

    if total_tokens == 0:
        return 0, 0, 0.0, 0.0, 0.0

    counts = Counter(words)
    unique_types = len(counts)

    ttr = unique_types / total_tokens
    repetition_ratio = sum(freq for freq in counts.values() if freq > 1) / total_tokens
    hapax_ratio = sum(1 for freq in counts.values() if freq == 1) / unique_types

    return total_tokens, unique_types, ttr, repetition_ratio, hapax_ratio


def hdd_calc(words, sample_size=42):
    N = len(words)
    if N == 0:
        return 0.0

    if N < sample_size:
        sample_size = N

    freqs = Counter(words)

    def comb(n, k):
        return math.comb(n, k) if 0 <= k <= n else 0

    D = 0.0
    for f in freqs.values():
        if N - f >= sample_size:
            D += 1.0 - comb(N - f, sample_size) / comb(N, sample_size)
        else:
            D += 1.0

    return D / sample_size


def Lex_parameters(prsnl_ID, task_nmbr, sample_size=42):
    base_dir = r"E:/diplomka/DLBNLP"
    file_name = prsnl_ID + "_CZ-AZV-TSK" + task_nmbr + "_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, file_name)

    model = whisper.load_model("small")
    result = model.transcribe(wav_path, language="cs")
    transcript = result["text"]

    total_tokens, unique_types, ttr, repetition_ratio, hapax_ratio = analyze_text(transcript)
    words = tokenize(transcript)
    hdd_value = hdd_calc(words, sample_size=sample_size)

    return total_tokens, unique_types, ttr, repetition_ratio, hapax_ratio, hdd_value, transcript

# Semanticke parametry ########################################################

nlp = stanza.Pipeline("cs", processors="tokenize,mwt,pos", tokenize_no_ssplit=True)


def Content_density(text: str):
    doc = nlp(text)
    tokens = []
    content_tokens = []

    content_pos = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}

    for sent in doc.sentences:
        for word in sent.words:
            if word.upos != "PUNCT":
                tokens.append(word)
                if word.upos in content_pos:
                    content_tokens.append(word)

    if len(tokens) == 0:
        return 0.0, 0, 0

    density = len(content_tokens) / len(tokens)
    return float(density), len(content_tokens), len(tokens)


def Semantic_parameters(text: str):
    density, n_content, n_tokens = Content_density(text)

    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return (
            0.0, 0.0, 0.0, 0, 0, 0,
            density, n_content, n_tokens
        )

    personal_pronouns = {
        "já", "mně", "mě", "mi", "mnou",
        "jáma", "jásem",
        "my", "nás", "nám", "námi",
        "náma"
    }

    vague_words = {
        "věc", "věci", "věcí", "něco", "cosi",
        "nějak", "jaksi", "jakoby", "jako",
        "prostě", "vlastně", "teda", "tedy", "no",
        "takový", "takovej", "taková", "takové",
        "tohle", "tomhle", "tadyto", "tamto",
    }

    categories = {
        "people": {
            "člověk", "lidi", "lidí", "osoba", "osoby",
            "rodina", "rodiny", "manželka", "manžel", "partner", "partnerka",
            "maminka", "matka", "máma", "otec", "táta", "tatínek",
            "bratr", "brácha", "sestra", "sestřenice", "bratranec",
            "syn", "dcera", "vnuk", "vnučka", "děda", "babička",
            "kluk", "chlapeček", "holka", "dítě", "děti",
            "kamarád", "kamarádka", "soused", "sousedka",
            "doktor", "sestra", "pečovatelka", "ošetřovatel",
            "navštěva", "návštěva", "host", "hosté",
        },
        "actions": {
            "jít", "jdu", "šel", "šla", "šli", "chodit", "chodím",
            "přijít", "přišel", "přišla", "přišli",
            "odejít", "odešel", "odešla",
            "bavit", "bavili", "mluvit", "mluvím", "mluvil",
            "říkat", "říkám", "řekl", "řekla",
            "poslouchat", "poslouchal", "dívat", "dívám", "díval",
            "jíst", "jedl", "pít", "pil", "popíjet",
            "dělat", "udělat", "pracovat", "vařit", "číst", "spát",
            "zůstat", "zůstal", "přijít", "dojít", "vrátit", "vrátil",
            "sedět", "stát", "ležet", "běhat", "pohnout", "nosit",
            "pomáhat", "pomohl", "starat", "starám",
            "vzít", "vzala", "vzali", "dát", "dal", "dali",
        },
        "objects": {
            "čaj", "kafe", "kávu", "káva", "chleba", "rohlík", "rohlíky", "pečivo",
            "jídlo", "pití", "polévku", "polévka", "oběd", "večeře", "večeři", "svačina",
            "svačinu","televize", "televizi", "tv", "rádio", "hudba", "hudbu", "mobil", "mobilu", "telefon", "telefonu",
            "kniha", "knihu", "noviny", "časopis",
            "stůl", "stolu", "židle", "židli", "postel", "posteli", "pohovka", "pohovce", "gauč", "gauči", "skříň", "skříni",
            "dům", "byt", "pokoj",
            "auto", "autem", "vlak", "vlakem", "autobus", "autobusem",
            "oblečení", "tričko", "kalhoty", "boty", "kabát",
            "hračka", "hračku", "hračky", "míč"
        },
        "time": {
            "ráno", "dopoledne", "odpoledne", "večer", "noc",
            "včera", "dnes", "dneska", "zítra", "později", "hned", "teď",
            "předtím", "potom", "pak", "mezi", "mezitím",
            "hodina", "hodiny", "minuta", "minuty",
            "den", "dny", "týden", "týdny", "měsíc", "rok", "roky",
        },
        "place": {
            "doma", "domů", "byt", "dům", "pokoj",
            "venku", "uvnitř", "tady", "tam",
            "kuchyně", "kuchyni", "obývák", "ložnice", "zahrada",
            "město", "vesnice", "ulice", "obchod", "kostel",
            "doktor", "nemocnice", "ordinace",
            "zastávka",
        },
    }

    personal_count = sum(1 for w in tokens if w in personal_pronouns)
    vague_count = sum(1 for w in tokens if w in vague_words)

    used_categories = set()
    for w in tokens:
        for cat_name, cat_words in categories.items():
            if w in cat_words:
                used_categories.add(cat_name)

    semantic_category_count = len(used_categories)

    personal_ratio = personal_count / total_tokens
    vague_ratio = vague_count / total_tokens

    return (
        personal_ratio,
        vague_ratio,
        personal_count,
        vague_count,
        semantic_category_count,
        density,
        n_content,
        n_tokens,
    )

# Syntakticke parametry ########################################################

def sentence_split_with_end(text: str):
    sentences = []
    buffer = []
    for ch in text:
        buffer.append(ch)
        if ch == ".":
            s = "".join(buffer).strip()
            if s:
                sentences.append((s, ch))
            buffer = []
    rest = "".join(buffer).strip()
    if rest:
        sentences.append((rest, rest[-1]))
    return sentences


CONJUCTIONS = {
     "a", "i", "ani", "nebo", "též", "rovněž", "dokonce",
     "a to", "ani", "ale", "avšak",
     "však", "nicméně", "nýbrž", "jenže", "leč", "anebo",
     "či", "neboť", "vždyť", "totiž", "proto", "a proto",
     "a tak", "tedy", "tudíž", "tak", "že", "co", "jak", "kdo", "který",
     "čí", "jaký", "jenž", "kde", "kdy", "když", "zatímco",
     "jakmile", "dokud", "než", "až", "sotva", "jen co", "kam",
     "odkud", "kudy", "jako", "jako by", "čím", "tím", "než aby",
     "protože", "jelikož", "poněvadž", "aby", "jestliže", "kdyby",
     "ačkoli", "ač", "třebaže", "přestože", "i když", "byť", "takže", "pak"
}


def count_conjucted_clauses(tokens):
    c = 0
    i = 0
    while i < len(tokens):
        if i+1 < len(tokens) and tokens[i] == "i" and tokens[i+1] == "když":
            c += 1
            i += 2
            continue
        if tokens[i] in CONJUCTIONS:
            c += 1
        i += 1
    return c


def count_hesitations(text: str):
    return text.count("...")


def Syntax_parameters(text: str, prsnl_ID: str, task_nmbr: str):
    base_dir = r"E:/diplomka/DLBNLP"
    wav_name = f"{prsnl_ID}_CZ-AZV-TSK{task_nmbr}_1.wav"
    wav_path = os.path.join(base_dir, prsnl_ID, wav_name)

    snd = parselmouth.Sound(wav_path)
    duration_seconds = snd.duration
    minutes = duration_seconds / 60.0

    sent_with_end = sentence_split_with_end(text)
    if not sent_with_end:
        return 0, 0.0, 0.0, 0, 0, 0.0

    sentences = [s for s, end in sent_with_end]
    tokenized = [tokenize(s) for s in sentences]
    lengths = [len(t) for t in tokenized if len(t) > 0]

    num_sent = len(lengths)
    mlu = mean(lengths) if lengths else 0.0
    total_sub = sum(count_conjucted_clauses(t) for t in tokenized)
    sub_per_sent = total_sub / num_sent if num_sent > 0 else 0.0

    hes = count_hesitations(text)
    hes_per_min = hes / minutes if minutes > 0 else 0.0

    return num_sent, mlu, sub_per_sent, total_sub, hes, hes_per_min
