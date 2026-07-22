# -*- coding: utf-8 -*-
# ============================================================
#  Muraqqam Challenge — v17-LITE (نسخة خفيفة: ~1.5-2 ساعة على T4)
#  لا يحتاج artifacts.npz — يبنيها بنفسه ويحفظها في Drive أولًا بأول.
#  المطلوب فقط: train.csv و test.csv
#
#  الخطة (وصفة v15c مختصرة):
#    1) S  = AraBERT-base   (4 طيات، 6 حقب) بدون تسميات زائفة
#    2) تسميات زائفة من مزيج S (ثقة >= 0.90)
#    3) CA = CAMeLBERT-MSA  (4 طيات، 6 حقب) + التسميات الزائفة
#    4) مزيج بأوزان معايرة -> معايرة معاملات -> محكمة ﷺ + قاعدة النهاية
#    5) حفظ artifacts.npz (لإضافة مكونات لاحقة بسكربت v16) + submission.csv
#
#  المدة التقريبية على T4 مفرد: ~1.5-2 ساعة (النموذج الكبير SL معطل افتراضيا:
#    TRAIN_SL=True يضيفه لاحقا +~2.5س — الكاش يحفظ ما اكتمل فلا شيء يعاد)
#  مهم على Kaggle: شغلي بـ Save & Run All (وليس تفاعليا) حتى لا تموت الجلسة.
#  لو انقطعت الجلسة: أعيدي التشغيل وسيستأنف من آخر طية محفوظة.
# ============================================================

import os, random, gc, shutil, inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import Counter

BASE_SEED = 42
random.seed(BASE_SEED); np.random.seed(BASE_SEED); torch.manual_seed(BASE_SEED)

# ===== إعدادات =====
ON_KAGGLE = os.path.exists('/kaggle/working')   # كاقل أولا — بعض أجهزته فيها /content أيضا
ON_COLAB  = (not ON_KAGGLE) and os.path.exists('/content')
WORK_DIR = '/kaggle/working' if ON_KAGGLE else ('/content/working' if ON_COLAB else './working')
os.makedirs(WORK_DIR, exist_ok=True)
CKPT_DIR = os.path.join(WORK_DIR, 'ckpt_tmp')

FAST_RUN = False          # True = تجربة سريعة فقط (ليست للتسليم)
NUM_EPOCHS   = 1 if FAST_RUN else 6
NUM_EPOCHS_L = 1 if FAST_RUN else 6   # الكبير يشبع مبكرا
NF           = 2 if FAST_RUN else 4   # طيات لكل عائلة
CONF = 0.90
TRAIN_SL = False   # True لإضافة AraBERT-large (+~2.5 ساعة)

MODEL_S  = 'aubmindlab/bert-base-arabertv02'
MODEL_CA = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
MODEL_L  = 'aubmindlab/bert-large-arabertv02'

# ===== ربط Google Drive على Colab (حفظ دائم لا يفنى بموت الجلسة) =====
DRIVE_DIR = None
if ON_COLAB:
    try:
        from google.colab import drive as _gdrive
        _gdrive.mount('/content/drive')
        DRIVE_DIR = '/content/drive/MyDrive/muraqqam'
        os.makedirs(DRIVE_DIR, exist_ok=True)
        print('Drive جاهز:', DRIVE_DIR)
    except Exception as _e:
        print('تحذير: لم يتم ربط Drive — الحفظ محلي فقط:', _e)
# على Kaggle: الكاش داخل /kaggle/working ويُحفظ مع مخرجات النسخة
CACHE_ROOT = DRIVE_DIR if DRIVE_DIR else WORK_DIR

def _sync_to_drive():
    if not DRIVE_DIR: return
    try:
        for fn in ('results.txt', 'submission.csv', 'artifacts.npz'):
            p = os.path.join(WORK_DIR, fn)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(DRIVE_DIR, fn))
    except Exception:
        pass

RESULTS = []
def log(msg):
    print(msg)
    RESULTS.append(str(msg))
    with open(os.path.join(WORK_DIR, 'results.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(RESULTS))
    _sync_to_drive()

log(f'GPU متاح: {torch.cuda.is_available()} | الجهاز: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
if FAST_RUN: log('FAST_RUN=True: تشغيل تجريبي، ليس للتسليم')

# ---------- 0) كود المقياس الرسمي ----------
VALID_SYMBOLS = set('.،؟!:؛-')
SYMBOLS = sorted(VALID_SYMBOLS)

def _tokenize_gold(text):
    leading, pairs, cur, in_word = [], [], [], False
    for ch in text:
        if ch.isspace():
            if in_word: pairs.append([''.join(cur), []]); cur=[]; in_word=False
            continue
        if ch in VALID_SYMBOLS:
            if in_word: pairs.append([''.join(cur), [ch]]); cur=[]; in_word=False
            else:
                if pairs: pairs[-1][1].append(ch)
                else: leading.append(ch)
            continue
        if not in_word: in_word=True; cur=[ch]
        else: cur.append(ch)
    if in_word: pairs.append([''.join(cur), []])
    return ''.join(leading), [(w, ''.join(g)) for (w,g) in pairs]

def _extract_labels(raw_text, generated_text, role):
    raw_words = str(raw_text).strip().split()
    if not raw_words: raise ValueError(f'[{role}] no words')
    _, pairs = _tokenize_gold(str(generated_text))
    gen_words = [w for (w,_) in pairs]
    if len(gen_words) != len(raw_words):
        raise ValueError(f'[{role}] word-count mismatch')
    for rw, gw in zip(raw_words, gen_words):
        if rw != gw: raise ValueError(f'[{role}] word mismatch')
    return [([c for c in gap if c in VALID_SYMBOLS] or ['0']) for _,gap in pairs]

# ---------- 1) الملفات ----------
def find_file(filename):
    roots = ['/content', '/kaggle/input', '.']
    skips = [os.path.abspath(WORK_DIR), os.path.abspath('/content/drive')]
    matches = []
    for root in roots:
        if not os.path.exists(root): continue
        for dirname, dirnames, filenames in os.walk(root):
            ad = os.path.abspath(dirname)
            if any(ad.startswith(s) for s in skips):
                dirnames[:] = []; continue
            if filename in filenames:
                matches.append(os.path.join(dirname, filename))
    return sorted(matches, key=lambda p: (len(p), p))

train_path = (find_file('train.csv') or [None])[0]
test_path  = (find_file('test.csv') or [None])[0]
assert train_path and test_path, '!! ضعي train.csv و test.csv'
log(f'train: {train_path}\ntest:  {test_path}')
train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

PBUH = 'ﷺ'
KEEP = ['', '،', '.', ':', '-', '؛', '!', '؟', '؟!', '!-', '-:']
LBL2ID = {l: i for i, l in enumerate(KEEP)}
MAP_RARE = {':-': '-:', '!-:': '!-', '؟-': '؟', '-؛': '؛', '؟!-': '؟!',
            '-،': '،', '-.': '.', '.،': '.', '!.': '!'}
LBL_MATRIX = np.zeros((len(KEEP), len(SYMBOLS)), dtype=np.int8)
for li, l in enumerate(KEEP):
    for si, s in enumerate(SYMBOLS):
        if s in l: LBL_MATRIX[li, si] = 1

def dedup_gap(g):
    seen = []
    for c in g:
        if c != '0' and c not in seen: seen.append(c)
    return ''.join(seen)

def norm_label(l):
    if l in LBL2ID: return l
    if l in MAP_RARE: return MAP_RARE[l]
    for c in l:
        if c in LBL2ID: return c
    return ''

docs = []
for _, row in train_df.iterrows():
    try:
        gaps = _extract_labels(row['text'], row['final_text'], 'gold')
    except ValueError:
        continue
    words = str(row['text']).strip().split()
    docs.append({'words': words, 'labels': [norm_label(dedup_gap(g)) for g in gaps]})
log(f'محاذاة: {len(docs)}/{len(train_df)}')
label_counts = Counter(l for d in docs for l in d['labels'])
test_words_all = [str(t).strip().split() for t in test_df['text']]

gold_flat = np.concatenate([[LBL2ID[l] for l in d['labels']] for d in docs]).astype(np.int64)
gold_bin = LBL_MATRIX[gold_flat]

def macro_of_probs(probs_flat, scales=None):
    s = scales if scales is not None else np.ones(len(KEEP))
    pb = LBL_MATRIX[np.argmax(probs_flat * s[None, :], axis=1)]
    f1s = []
    for si in range(len(SYMBOLS)):
        tp = int(((gold_bin[:,si]==1)&(pb[:,si]==1)).sum())
        fp = int(((gold_bin[:,si]==0)&(pb[:,si]==1)).sum())
        fn = int(((gold_bin[:,si]==1)&(pb[:,si]==0)).sum())
        p_ = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
    return float(np.mean(f1s))

# ---------- 2) مكونات التدريب ----------
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          TrainingArguments, Trainer, DataCollatorForTokenClassification)

MAX_WORDS = 200
STRIDE    = 100

def make_windows(n):
    if n <= MAX_WORDS: return [(0, n)]
    wins, s = [], 0
    while True:
        e = min(s + MAX_WORDS, n)
        wins.append((s, e))
        if e == n: break
        s += STRIDE
    return wins

tokenizer = None

class PunctDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        words, labels = self.items[idx]
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=512)
        word_ids = enc.word_ids()
        last_sub = {}
        for pos, wid in enumerate(word_ids):
            if wid is not None: last_sub[wid] = pos
        lab = [-100] * len(word_ids)
        for wid, pos in last_sub.items():
            if labels[wid] is not None:
                lab[pos] = LBL2ID[labels[wid]]
        enc['labels'] = lab
        return {k: torch.tensor(v) for k, v in enc.items()}

def docs_to_items(dd):
    items = []
    for d in dd:
        for s, e in make_windows(len(d['words'])):
            items.append((d['words'][s:e], d['labels'][s:e]))
    return items

w = np.array([1.0 / np.sqrt(max(label_counts.get(l, 1), 1)) for l in KEEP])
w = w / w.mean(); w[0] = min(w[0], 0.5)
class_weights = torch.tensor(w, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device), ignore_index=-100)
        loss = loss_fct(outputs.logits.view(-1, len(KEEP)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids
    mask = labels != -100
    yt = LBL_MATRIX[labels[mask]]; yp = LBL_MATRIX[preds[mask]]
    f1s = []
    for si in range(len(SYMBOLS)):
        tp = int(((yt[:,si]==1)&(yp[:,si]==1)).sum())
        fp = int(((yt[:,si]==0)&(yp[:,si]==1)).sum())
        fn = int(((yt[:,si]==1)&(yp[:,si]==0)).sum())
        p_ = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
    return {'macro_f1_7': float(np.mean(f1s))}

@torch.no_grad()
def predict_doc_probs(model, words, win_batch=8):
    """نفس منطق v15c (النافذة الأوسط تفوز) لكن بدفعات نوافذ — أسرع ~8 مرات."""
    n = len(words)
    best = [None] * n
    wins = make_windows(n)
    for i in range(0, len(wins), win_batch):
        chunk_wins = wins[i:i + win_batch]
        enc = tokenizer([words[s:e] for s, e in chunk_wins],
                        is_split_into_words=True, truncation=True,
                        max_length=512, padding=True, return_tensors='pt')
        word_ids_list = [enc.word_ids(batch_index=bi) for bi in range(len(chunk_wins))]
        enc = {k: v.to(model.device) for k, v in enc.items()}
        probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
        for bi, (s, e) in enumerate(chunk_wins):
            last_sub = {}
            for pos, wid in enumerate(word_ids_list[bi]):
                if wid is not None: last_sub[wid] = pos
            L = e - s
            for wid, pos in last_sub.items():
                gi = s + wid
                central = min(wid, L - 1 - wid)
                if best[gi] is None or central > best[gi][0]:
                    best[gi] = (central, probs[bi, pos])
    out = np.zeros((n, len(KEEP)), dtype=np.float32)
    for i, b in enumerate(best):
        if b is not None: out[i] = b[1]
        else: out[i, 0] = 1.0
    return out

def train_family(model_name, n_folds, seed_offset, tag, pseudo,
                 bs=16, lr=3e-5, ga=1, epochs=None):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rng = random.Random(BASE_SEED)
    order = list(range(len(docs))); rng.shuffle(order)
    folds = [order[i::n_folds] for i in range(n_folds)]
    oof = [None] * len(docs)
    tps = [np.zeros((len(ws), len(KEEP)), dtype=np.float32) for ws in test_words_all]
    epochs = epochs or NUM_EPOCHS
    cache_dir = os.path.join(CACHE_ROOT, 'foldcache_v17')
    os.makedirs(cache_dir, exist_ok=True)
    def _cache_path(f): return os.path.join(cache_dir, f'{tag}_f{f}.npz')
    for fold in range(n_folds):
        cp = _cache_path(fold)
        if os.path.exists(cp):
            C = np.load(cp)
            idxs = C['idx']; oflat = C['oof']; olens = C['olens']
            s = 0
            for di, L in zip(idxs, olens):
                oof[int(di)] = oflat[s:s+L].astype(np.float32); s += L
            tflat = C['test']; tlens = C['tlens']; s = 0
            for ti, L in enumerate(tlens):
                tps[ti] += tflat[s:s+L].astype(np.float32); s += L
            log(f'--- {tag} fold {fold+1}/{n_folds}: مستأنف من الكاش ✓ ---')
            continue
        log(f'--- {tag} fold {fold+1}/{n_folds} ---')
        val_idx = set(folds[fold])
        tr = [docs[i] for i in range(len(docs)) if i not in val_idx] + pseudo
        va = [docs[i] for i in folds[fold]]
        torch.manual_seed(BASE_SEED+seed_offset+fold); np.random.seed(BASE_SEED+seed_offset+fold)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(KEEP))
        ta_kwargs = dict(
            output_dir=CKPT_DIR, num_train_epochs=epochs,
            per_device_train_batch_size=bs, per_device_eval_batch_size=32,
            gradient_accumulation_steps=ga,
            learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
            save_strategy='epoch', save_total_limit=1,
            load_best_model_at_end=True, metric_for_best_model='macro_f1_7',
            fp16=torch.cuda.is_available(), logging_steps=200, report_to='none',
            seed=BASE_SEED+seed_offset+fold)
        if 'eval_strategy' in inspect.signature(TrainingArguments.__init__).parameters:
            ta_kwargs['eval_strategy'] = 'epoch'
        else:
            ta_kwargs['evaluation_strategy'] = 'epoch'
        args = TrainingArguments(**ta_kwargs)
        trainer = WeightedTrainer(model=model, args=args,
                                  train_dataset=PunctDataset(docs_to_items(tr)),
                                  eval_dataset=PunctDataset(docs_to_items(va)),
                                  data_collator=DataCollatorForTokenClassification(tokenizer),
                                  compute_metrics=compute_metrics)
        trainer.train()
        ev = trainer.evaluate()
        log(f'{tag} fold {fold+1}: macro_f1_7={ev["eval_macro_f1_7"]:.4f}')
        model.eval()
        fold_test = []
        for di in folds[fold]:
            oof[di] = predict_doc_probs(model, docs[di]['words'])
        for ti, ws in enumerate(test_words_all):
            fp = predict_doc_probs(model, ws)
            tps[ti] += fp
            fold_test.append(fp)
        idxs = np.array(folds[fold], dtype=np.int32)
        oflat = np.concatenate([oof[int(di)] for di in idxs], axis=0).astype(np.float16)
        olens = np.array([oof[int(di)].shape[0] for di in idxs], dtype=np.int32)
        tflat = np.concatenate(fold_test, axis=0).astype(np.float16)
        tlens = np.array([f.shape[0] for f in fold_test], dtype=np.int32)
        np.savez_compressed(cp, idx=idxs, oof=oflat, olens=olens, test=tflat, tlens=tlens)
        log(f'حُفظت الطية {fold+1} في الكاش ({cache_dir})')
        del trainer, model
        gc.collect(); torch.cuda.empty_cache()
        shutil.rmtree(CKPT_DIR, ignore_errors=True)
    for ti in range(len(tps)):
        tps[ti] /= n_folds
    return oof, tps

# ---------- 3) المرحلة الأولى: عائلة S بدون تسميات زائفة ----------
oof_S, test_S = train_family(MODEL_S, NF, 100, 'S', pseudo=[], bs=16, lr=3e-5)
sc_S = macro_of_probs(np.concatenate(oof_S, axis=0))
log(f'S منفردة: OOF={sc_S:.4f}')

# ---------- 4) تسميات زائفة من مزيج S ----------
pseudo_docs, kept_pos, tot_pos = [], 0, 0
for ti, ws in enumerate(test_words_all):
    if not ws: continue
    probs = test_S[ti]
    ids = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(ids)), ids]
    labels = [KEEP[i] if c >= CONF else None for i, c in zip(ids, conf)]
    kept_pos += sum(l is not None for l in labels); tot_pos += len(labels)
    pseudo_docs.append({'words': ws, 'labels': labels})
log(f'تسميات زائفة: {kept_pos}/{tot_pos} ({kept_pos/tot_pos:.1%}) موثوقة')

# ---------- 5) المرحلة الثانية: CA و SL مع التسميات الزائفة ----------
oof_CA, test_CA = train_family(MODEL_CA, NF, 300, 'CA', pseudo=pseudo_docs, bs=16, lr=3e-5)
log(f'CA منفردة: OOF={macro_of_probs(np.concatenate(oof_CA, axis=0)):.4f}')

if TRAIN_SL:
    oof_SL, test_SL = train_family(MODEL_L, NF, 800, 'SL', pseudo=pseudo_docs,
                                   bs=8, lr=2e-5, ga=2, epochs=NUM_EPOCHS_L)
    log(f'SL منفردة: OOF={macro_of_probs(np.concatenate(oof_SL, axis=0)):.4f}')

# ---------- 6) مزيج متعدد المكونات ----------
names = ['S', 'CA'] + (['SL'] if TRAIN_SL else [])
family_oof  = {'S': oof_S, 'CA': oof_CA}
family_test = {'S': test_S, 'CA': test_CA}
if TRAIN_SL:
    family_oof['SL'] = oof_SL; family_test['SL'] = test_SL
comps_oof  = [np.concatenate(family_oof[n], axis=0) for n in names]
comps_test = [family_test[n] for n in names]

K = len(comps_oof)
fam_w = np.ones(K) / K
W_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
def mix_with(wv):
    wv = np.array(wv); s = wv.sum()
    if s == 0: return None
    return sum((w_/s)*c for w_, c in zip(wv, comps_oof))
best_sc = macro_of_probs(mix_with(fam_w))
for _pass in range(3):
    for fi in range(K):
        best_v = fam_w[fi]
        for v in W_GRID:
            trial = fam_w.copy(); trial[fi] = v
            mixed = mix_with(trial)
            if mixed is None: continue
            sc = macro_of_probs(mixed)
            if sc > best_sc: best_sc, best_v = sc, v
        fam_w[fi] = best_v
fam_w = fam_w / fam_w.sum()
log('أوزان المزج: ' + str({n: round(float(w_),3) for n, w_ in zip(names, fam_w)}) + f' -> OOF={best_sc:.4f}')
oof_flat = mix_with(fam_w)

# ---------- 7) معايرة + محكمة ﷺ ----------
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.1, 2.5, 3.0]
scales = np.ones(len(KEEP), dtype=np.float32)
cur = macro_of_probs(oof_flat, scales)
for _pass in range(3):
    for ci in range(1, len(KEEP)):
        best_v = scales[ci]
        for v in GRID:
            scales[ci] = v
            sc = macro_of_probs(oof_flat, scales)
            if sc > cur: cur, best_v = sc, v
        scales[ci] = best_v
log(f'OOF بعد المعايرة: {cur:.4f}')
log('المعاملات: ' + str({l: round(float(s),2) for l,s in zip(KEEP, scales)}))

words_flat = np.concatenate([np.array(d['words']) for d in docs])
doc_of = np.concatenate([np.full(len(d['words']), i) for i, d in enumerate(docs)])
is_last_flat = np.concatenate([np.array([j == len(d['words'])-1 for j in range(len(d['words']))])
                               for d in docs])
is_pbuh = words_flat == PBUH
pbuh_next = np.zeros(len(words_flat), dtype=bool)
pbuh_next[:-1] = is_pbuh[1:] & (doc_of[:-1] == doc_of[1:])

TERMINAL_IDS = {LBL2ID[l] for l in KEEP if any(c in l for c in '.!؟')}
DASH_IDS = {i for i,l in enumerate(KEEP) if '-' in l}
ID_DASH = LBL2ID['-']; ID_DOT = LBL2ID['.']

def macro_ids(pred_ids):
    pb = LBL_MATRIX[pred_ids]
    f1s = []
    for si in range(len(SYMBOLS)):
        tp = int(((gold_bin[:,si]==1)&(pb[:,si]==1)).sum())
        fp = int(((gold_bin[:,si]==0)&(pb[:,si]==1)).sum())
        fn = int(((gold_bin[:,si]==1)&(pb[:,si]==0)).sum())
        p_ = tp/(tp+fp) if tp+fp else 0.0
        r_ = tp/(tp+fn) if tp+fn else 0.0
        f1s.append(2*p_*r_/(p_+r_) if p_+r_ else 0.0)
    return float(np.mean(f1s))

base_pred = np.argmax(oof_flat * scales[None, :], axis=1)

def apply_pbuh(pred, mode):
    pred = pred.copy()
    if mode == 'always':
        m = pbuh_next & ~np.isin(pred, list(DASH_IDS)); pred[m] = ID_DASH
        m = is_pbuh & ~np.isin(pred, list(DASH_IDS)); pred[m] = ID_DASH
    elif mode == 'soft':
        m = pbuh_next & (pred == 0); pred[m] = ID_DASH
        m = is_pbuh & (pred == 0); pred[m] = ID_DASH
    return pred

def apply_end(pred):
    pred = pred.copy()
    m = is_last_flat & ~np.isin(pred, list(TERMINAL_IDS))
    pred[m] = ID_DOT
    return pred

best_mode, best_msc = None, -1
for mode in ['off', 'soft', 'always']:
    sc = macro_ids(apply_end(apply_pbuh(base_pred, mode)))
    log(f'قاعدة ﷺ [{mode}]: OOF={sc:.4f}')
    if sc > best_msc: best_msc, best_mode = sc, mode
log(f'الوضع المختار: {best_mode} | OOF نهائي: {best_msc:.4f}')

# ---------- 8) حفظ الذخيرة الجديدة (متوافقة مع سكربت v16) ----------
def pack(doc_arrays):
    lens = np.array([a.shape[0] for a in doc_arrays], dtype=np.int32)
    return np.concatenate(doc_arrays, axis=0).astype(np.float16), lens

save_kw = dict(fam_w=fam_w.astype(np.float32), scales=scales.astype(np.float32),
               mode_idx=np.int32(['off','soft','always'].index(best_mode)),
               comp_names=np.array(names))
KEYMAP = {'S': 'ar2', 'CA': 'ca1', 'SL': 'arL'}
for nm in names:
    key = KEYMAP[nm]
    p, l = pack(family_oof[nm]);  save_kw[f'oof_{key}'] = p;  save_kw[f'oof_{key}_len'] = l
    p, l = pack(family_test[nm]); save_kw[f'test_{key}'] = p; save_kw[f'test_{key}_len'] = l
np.savez_compressed(os.path.join(WORK_DIR, 'artifacts.npz'), **save_kw)
log('حُفظت ذخيرة v17 (artifacts.npz)')

# ---------- 9) بناء التسليم ----------
finals = []
for ti, ws in enumerate(test_words_all):
    if not ws:
        finals.append(''); continue
    probs = sum(w_*c[ti] for w_, c in zip(fam_w, comps_test))
    pred = np.argmax(probs * scales[None, :], axis=1)
    w_arr = np.array(ws)
    pb = w_arr == PBUH
    pbn = np.zeros(len(ws), dtype=bool); pbn[:-1] = pb[1:]
    if best_mode == 'always':
        m = pbn & ~np.isin(pred, list(DASH_IDS)); pred[m] = ID_DASH
        m = pb & ~np.isin(pred, list(DASH_IDS)); pred[m] = ID_DASH
    elif best_mode == 'soft':
        m = pbn & (pred == 0); pred[m] = ID_DASH
        m = pb & (pred == 0); pred[m] = ID_DASH
    if pred[-1] not in TERMINAL_IDS: pred[-1] = ID_DOT
    finals.append(' '.join(t + KEEP[p] for t, p in zip(ws, pred)))

bad = []
for i, (raw, fin) in enumerate(zip(test_df['text'], finals)):
    try:
        _extract_labels(raw, fin, 'prediction')
    except ValueError as e:
        bad.append((i, str(e)))
if bad:
    log('!! فشل الفحص الذاتي: ' + str(bad[:5]))
    raise SystemExit('لا تُسلِّم')
log(f'الفحص الذاتي: كل الصفوف الـ {len(finals)} صالحة')

sub = pd.DataFrame({'id': test_df['id'], 'final_text': finals})
sub.to_csv(os.path.join(WORK_DIR, 'submission.csv'), index=False)
log('تم حفظ submission.csv في ' + WORK_DIR)
_sync_to_drive()
if DRIVE_DIR:
    log('كل المخرجات منسوخة إلى Google Drive: ' + DRIVE_DIR)
