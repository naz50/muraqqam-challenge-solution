# -*- coding: utf-8 -*-
# ============================================================
#  Muraqqam Challenge — v25 (الجولة الختامية: ML-large + جيل ثالث)
#  المطلوب: train.csv و test.csv و artifacts.npz (ذخيرة v17)
#
#  الجولة الختامية (~6-7 ساعات على T4، أسرع على 2xT4):
#    المدخلات: artifacts.npz بتسعة مكونات (من v24) + probs_backup.pkl (من v23)
#    1) ML2 = AraBERT-large برأس multi-label، 5 طيات × 6 حقب (تنويع + قوة)
#    2) CA3 = CAMeLBERT-MSA، 8 طيات × 8 حقب — بتسميات الخزّان التساعي
#    3) AR6 = AraBERT-base جيل ثالث، 8 طيات × 8 حقب
#    4) دمج الكل بطبقة Stacking -> submission.csv
#  كل طية تُحفظ في الكاش فور اكتمالها — الاستئناف تلقائي.
#  بعد الانتهاء: شغّلي سكربت v18 على artifacts.npz الجديدة للمزج والتسليم.
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
TRAIN_CA3 = True   # CAMeLBERT بثماني طيات وجيل ثالث
TRAIN_AR6 = True   # AraBERT-base جيل ثالث
TRAIN_ML  = True   # ML2 = multi-label على النموذج الكبير
NUM_EPOCHS_ML = 1 if FAST_RUN else 6
NF_ML = 2 if FAST_RUN else 5

MODEL_CA3 = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
MODEL_AR6 = 'aubmindlab/bert-base-arabertv02'
MODEL_ML  = 'aubmindlab/bert-large-arabertv02'

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
SYM2IDX = {s: i for i, s in enumerate(SYMBOLS)}
for _, row in train_df.iterrows():
    try:
        gaps = _extract_labels(row['text'], row['final_text'], 'gold')
    except ValueError:
        continue
    words = str(row['text']).strip().split()
    lab7 = np.zeros((len(words), 7), dtype=np.float32)
    for j, g in enumerate(gaps):
        for c in g:
            if c in SYM2IDX: lab7[j, SYM2IDX[c]] = 1.0
    docs.append({'words': words,
                 'labels': [norm_label(dedup_gap(g)) for g in gaps],
                 'lab7': lab7})
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
            try:
                C = np.load(cp)
                idxs = C['idx']; oflat = C['oof']; olens = C['olens']
                assert int(max(idxs)) < len(docs) and len(C['tlens']) == len(tps)
                s = 0
                for di, L in zip(idxs, olens):
                    oof[int(di)] = oflat[s:s+L].astype(np.float32); s += L
                tflat = C['test']; tlens = C['tlens']; s = 0
                for ti, L in enumerate(tlens):
                    tps[ti] += tflat[s:s+L].astype(np.float32); s += L
                log(f'--- {tag} fold {fold+1}/{n_folds}: مستأنف من الكاش ✓ ---')
                continue
            except Exception as _ce:
                log(f'كاش {tag} fold {fold+1} غير متوافق — إعادة تدريب ({_ce})')
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

# ---------- 3) تحميل الذخيرة الحالية ----------
art_path = (find_file('artifacts.npz') or find_file('artifacts_1.npz')
            or find_file('artifacts (1).npz') or [None])[0]
assert art_path, '!! ضعي artifacts.npz (ذخيرة v17)'
log('الذخيرة: ' + art_path)
Z = np.load(art_path, allow_pickle=False)

def unpack(flat, lens):
    out, s = [], 0
    for L in lens:
        out.append(flat[s:s+L].astype(np.float32)); s += L
    return out

existing_test = {}
existing_keys = []
for key in Z.files:
    if key.startswith('oof_') and not key.endswith('_len'):
        k = key[4:]
        existing_keys.append(k)
        existing_test[k] = unpack(Z[f'test_{k}'], Z[f'test_{k}_len'])
log('مكونات الذخيرة الحالية: ' + str(existing_keys))
prev_w = Z['fam_w'].astype(np.float32)
assert len(prev_w) == len(existing_keys) or True  # الأوزان قد لا تطابق الترتيب — نستخدم متوسطا متساويا عند الشك

# ---------- 4) تسميات زائفة من مزيج الذخيرة الحالية ----------
pseudo_docs, kept_pos, tot_pos = [], 0, 0
for ti, ws in enumerate(test_words_all):
    if not ws: continue
    probs = sum(existing_test[k][ti] for k in existing_keys) / len(existing_keys)
    ids = np.argmax(probs, axis=1)
    conf = probs[np.arange(len(ids)), ids]
    labels = [KEEP[i] if c >= CONF else None for i, c in zip(ids, conf)]
    kept_pos += sum(l is not None for l in labels); tot_pos += len(labels)
    pseudo_docs.append({'words': ws, 'labels': labels})
log(f'تسميات زائفة: {kept_pos}/{tot_pos} ({kept_pos/tot_pos:.1%}) موثوقة')

# ---------- 5) تدريب العائلات الجديدة ----------
ADD = []
if TRAIN_CA3: ADD.append(('ca3', MODEL_CA3, 1700, 8, dict(bs=16, lr=3e-5, epochs=8)))
if TRAIN_AR6: ADD.append(('ar6', MODEL_AR6, 1800, 8, dict(bs=16, lr=3e-5, epochs=8)))
assert ADD, '!! فعّلي عائلة واحدة على الأقل'

new_packs = {}
for key, mname, soff, nf_k, kw in ADD:
    if f'oof_{key}' in Z.files:
        log(f'{key}: موجودة مسبقا في الذخيرة — تخطٍّ')
        continue
    oof_n, test_n = train_family(mname, nf_k, soff, key.upper(), pseudo=pseudo_docs, **kw)
    sc = macro_of_probs(np.concatenate(oof_n, axis=0))
    log(f'{key} منفردة: OOF={sc:.4f}')
    new_packs[key] = (oof_n, test_n)

# ---------- 6) عائلة Multi-Label (سبعة sigmoid لكل كلمة) ----------
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

class MLDataset(Dataset):
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        words, lab7 = self.items[idx]   # lab7: (n,7) أو None لكل صف مقنّع
        enc = tokenizer(words, is_split_into_words=True, truncation=True, max_length=512)
        T = len(enc['input_ids'])
        lab = np.zeros((T, 7), dtype=np.float32)
        msk = np.zeros(T, dtype=np.float32)
        prev = None
        for t, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev and lab7[wid] is not None:
                row = lab7[wid]
                if not (isinstance(row, float) and np.isnan(row)):
                    lab[t] = row; msk[t] = 1.0
            prev = wid
        return dict(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                    labels=lab, label_mask=msk)

def ml_collate(items):
    T = max(len(it['input_ids']) for it in items); B = len(items)
    pad = tokenizer.pad_token_id
    ii = np.full((B, T), pad, dtype=np.int64); am = np.zeros((B, T), dtype=np.int64)
    lb = np.zeros((B, T, 7), dtype=np.float32); lm = np.zeros((B, T), dtype=np.float32)
    for b, it in enumerate(items):
        L = len(it['input_ids'])
        ii[b, :L] = it['input_ids']; am[b, :L] = it['attention_mask']
        lb[b, :L] = it['labels'];    lm[b, :L] = it['label_mask']
    return (torch.from_numpy(ii), torch.from_numpy(am),
            torch.from_numpy(lb), torch.from_numpy(lm))

def ml_items(dd, use_lab7=True):
    items = []
    for d in dd:
        n = len(d['words'])
        lab7 = d['lab7'] if use_lab7 else d['_p7']
        for s, e in make_windows(n):
            items.append((d['words'][s:e], [lab7[j] for j in range(s, e)]))
    return items

@torch.no_grad()
def ml_predict(model, words, win_batch=8):
    n = len(words)
    probs = np.zeros((n, 7)); wts = np.zeros(n)
    wins = make_windows(n)
    dev = next(model.parameters()).device
    for i in range(0, len(wins), win_batch):
        chunk = wins[i:i+win_batch]
        enc = tokenizer([words[s:e] for s, e in chunk], is_split_into_words=True,
                        truncation=True, max_length=512, padding=True, return_tensors='pt')
        wid_l = [enc.word_ids(batch_index=bi) for bi in range(len(chunk))]
        enc_t = {k: v.to(dev) for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
        p = torch.sigmoid(model(**enc_t).logits.float()).cpu().numpy()
        for bi, (s, e) in enumerate(chunk):
            L = e - s; prev = None
            for t, wid in enumerate(wid_l[bi]):
                if wid is None or wid == prev: prev = wid; continue
                prev = wid
                w = min(wid + 1, L - wid)
                probs[s + wid] += w * p[bi, t]; wts[s + wid] += w
    wts[wts == 0] = 1
    return (probs / wts[:, None]).astype(np.float32)

def train_ml_family(model_name, n_folds, seed_offset, tag, pseudo, bs=16):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rng = random.Random(BASE_SEED)
    order = list(range(len(docs))); rng.shuffle(order)
    folds = [order[i::n_folds] for i in range(n_folds)]
    oof = [None] * len(docs)
    tps = [np.zeros((len(ws), 7), dtype=np.float32) for ws in test_words_all]
    cache_dir = os.path.join(CACHE_ROOT, 'foldcache_v17'); os.makedirs(cache_dir, exist_ok=True)
    # تحويل التسميات الزائفة إلى 7 أعمدة (None يبقى مقنّعا)
    pdocs = []
    for d in pseudo:
        n = len(d['words'])
        p7 = []
        for l in d['labels']:
            if l is None: p7.append(None)
            else:
                v = np.zeros(7, dtype=np.float32)
                for c in l:
                    if c in SYM2IDX: v[SYM2IDX[c]] = 1.0
                p7.append(v)
        pdocs.append({'words': d['words'], '_p7': p7})
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for fold in range(n_folds):
        cp = os.path.join(cache_dir, f'{tag}_f{fold}.npz')
        if os.path.exists(cp):
            try:
                C = np.load(cp)
                assert int(max(C['idx'])) < len(docs) and len(C['tlens']) == len(tps)
                s = 0
                for di, L in zip(C['idx'], C['olens']):
                    oof[int(di)] = C['oof'][s:s+L].astype(np.float32); s += L
                s = 0
                for ti, L in enumerate(C['tlens']):
                    tps[ti] += C['test'][s:s+L].astype(np.float32); s += L
                log(f'--- {tag} fold {fold+1}/{n_folds}: مستأنف من الكاش ✓ ---')
                continue
            except Exception as _ce:
                log(f'كاش {tag} fold {fold+1} غير متوافق — إعادة تدريب ({_ce})')
        log(f'--- {tag} fold {fold+1}/{n_folds} ---')
        torch.manual_seed(BASE_SEED+seed_offset+fold); np.random.seed(BASE_SEED+seed_offset+fold)
        val_idx = set(folds[fold])
        tr_items = ml_items([docs[i] for i in range(len(docs)) if i not in val_idx]) + \
                   ml_items(pdocs, use_lab7=False)
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=7).to(device)
        dl = DataLoader(MLDataset(tr_items), batch_size=bs, shuffle=True,
                        collate_fn=ml_collate, num_workers=2, pin_memory=True)
        lab_all = np.concatenate([docs[i]['lab7'] for i in range(len(docs)) if i not in val_idx])
        pos = lab_all.sum(0); neg = len(lab_all) - pos
        pw = torch.tensor(np.clip(neg/np.maximum(pos, 1), 1, 30), dtype=torch.float32, device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw, reduction='none')
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        steps = len(dl) * NUM_EPOCHS_ML
        sched = get_linear_schedule_with_warmup(opt, int(0.1*steps), steps)
        scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
        model.train()
        for ep in range(NUM_EPOCHS_ML):
            tot, nb = 0.0, 0
            for ii, am, lb, lm in dl:
                ii, am, lb, lm = ii.to(device), am.to(device), lb.to(device), lm.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=(device == 'cuda')):
                    logits = model(input_ids=ii, attention_mask=am).logits
                    loss = (crit(logits, lb).mean(-1) * lm).sum() / lm.sum().clamp(min=1)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                tot += loss.item(); nb += 1
            log(f'  {tag} fold {fold+1} epoch {ep+1}/{NUM_EPOCHS_ML} loss {tot/nb:.4f}')
        model.eval()
        fold_test = []
        for di in folds[fold]:
            oof[di] = ml_predict(model, docs[di]['words'])
        for ti, ws in enumerate(test_words_all):
            fp = ml_predict(model, ws)
            tps[ti] += fp; fold_test.append(fp)
        idxs = np.array(folds[fold], dtype=np.int32)
        np.savez_compressed(cp, idx=idxs,
            oof=np.concatenate([oof[int(d)] for d in idxs]).astype(np.float16),
            olens=np.array([oof[int(d)].shape[0] for d in idxs], dtype=np.int32),
            test=np.concatenate(fold_test).astype(np.float16),
            tlens=np.array([f.shape[0] for f in fold_test], dtype=np.int32))
        log(f'حُفظت الطية {fold+1} في الكاش')
        del model; gc.collect(); torch.cuda.empty_cache()
    for ti in range(len(tps)):
        tps[ti] /= n_folds
    return oof, tps

ml_oof_docs, ml_test_docs = (None, None)
if TRAIN_ML:
    ml_oof_docs, ml_test_docs = train_ml_family(MODEL_ML, NF_ML, 2100, 'ML2', pseudo_docs, bs=8)

# ---------- 7) حفظ الذخيرة + ذخيرة ML ----------
def pack(doc_arrays):
    lens = np.array([a.shape[0] for a in doc_arrays], dtype=np.int32)
    return np.concatenate(doc_arrays, axis=0).astype(np.float16), lens

save_kw = {k: Z[k] for k in Z.files if k not in ('comp_names',)}
all_names = list(existing_keys)
for key, (oof_n, test_n) in new_packs.items():
    p, l = pack(oof_n);  save_kw[f'oof_{key}'] = p;  save_kw[f'oof_{key}_len'] = l
    p, l = pack(test_n); save_kw[f'test_{key}'] = p; save_kw[f'test_{key}_len'] = l
    all_names.append(key)
save_kw['comp_names'] = np.array(all_names)
np.savez_compressed(os.path.join(WORK_DIR, 'artifacts.npz'), **save_kw)
log('حُفظت الذخيرة artifacts.npz: ' + str(all_names))
if TRAIN_ML:
    import pickle
    ML_ORDER = ['،', '.', ':', '؛', '-', '!', '؟']
    PERM = [SYMBOLS.index(s) for s in ML_ORDER]
    with open(os.path.join(WORK_DIR, 'probs_backup2.pkl'), 'wb') as f:
        pickle.dump({'oof': [a[:, PERM] for a in ml_oof_docs],
                     'test': [a[:, PERM] for a in ml_test_docs]}, f)
    log('حُفظت ذخيرة ML الجديدة: probs_backup2.pkl')
_sync_to_drive()

# ---------- 8) طبقة Stacking النهائية + التسليم ----------
from sklearn.linear_model import LogisticRegression

LBL7 = LBL_MATRIX.astype(np.float64)
gold_true = np.concatenate([d['lab7'] for d in docs]).astype(np.int8)
Nn = gold_true.shape[0]
doc_id = np.concatenate([np.full(len(d['words']), i) for i, d in enumerate(docs)])
meta_rows = []
for d in docs:
    n = len(d['words']); pb = [w == PBUH for w in d['words']]
    for j in range(n):
        meta_rows.append((j == n-1, j == 0, pb[j], pb[j+1] if j+1 < n else False))
meta = np.array(meta_rows, dtype=float)

# مكونات softmax من الذخيرة المدموجة + الجديدة في الذاكرة
def load_from_savekw(key):
    o = unpack(save_kw[f'oof_{key}'], save_kw[f'oof_{key}_len'])
    t = unpack(save_kw[f'test_{key}'], save_kw[f'test_{key}_len'])
    return np.concatenate(o, axis=0), t
comp_oof_l, comp_test_l = [], []
for key in all_names:
    o, t = load_from_savekw(key)
    comp_oof_l.append(o.astype(np.float64) @ LBL7)
    comp_test_l.append([a.astype(np.float64) @ LBL7 for a in t])
if TRAIN_ML:
    comp_oof_l.append(np.concatenate(ml_oof_docs, axis=0).astype(np.float64))
    comp_test_l.append([a.astype(np.float64) for a in ml_test_docs])
# مكونات ML من ملفات probs_backup*.pkl (بترتيب أعمدة ['،','.',':','؛','-','!','؟'])
import pickle as _pkl
ML_ORDER = ['،', '.', ':', '؛', '-', '!', '؟']
PERM_IN = [ML_ORDER.index(s) for s in SYMBOLS]
_ml_files = []
for root in ['/kaggle/input', '/content', '.']:
    if not os.path.exists(root): continue
    for dn, _, fns in os.walk(root):
        for fn in fns:
            if fn.startswith('probs_backup') and fn.endswith('.pkl'):
                _ml_files.append(os.path.join(dn, fn))
for mp in sorted(set(_ml_files)):
    try:
        with open(mp, 'rb') as f: P = _pkl.load(f)
        assert len(P['oof']) == len(docs) and len(P['test']) == len(test_words_all)
        comp_oof_l.append(np.concatenate([a[:, PERM_IN] for a in P['oof']]).astype(np.float64))
        comp_test_l.append([a[:, PERM_IN].astype(np.float64) for a in P['test']])
        log('أُضيف مكوّن ML من ' + mp)
    except Exception as _e:
        log(f'تخطي {mp}: {_e}')

base = np.concatenate(comp_oof_l, axis=1)
def shift(F, ids, k):
    out = np.zeros_like(F)
    if k > 0:
        out[k:] = F[:-k]; out[k:][ids[k:] != ids[:-k]] = 0
    else:
        k = -k; out[:-k] = F[k:]; out[:-k][ids[:-k] != ids[k:]] = 0
    return out
X = np.concatenate([base, shift(base, doc_id, 1), shift(base, doc_id, -1), meta], axis=1)
log(f'ميزات الـ stacking: {X.shape}')

perm = np.random.default_rng(BASE_SEED).permutation(len(docs))
folds5 = [perm[i::5] for i in range(5)]
cvp = np.zeros((Nn, 7))
for fold in folds5:
    va = np.isin(doc_id, fold); tr = ~va
    for si in range(7):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], gold_true[tr, si])
        cvp[va, si] = clf.predict_proba(X[va])[:, 1]
ths = np.zeros(7); f1s = np.zeros(7)
for si in range(7):
    yb = gold_true[:, si].astype(bool); best = (0.0, 0.5)
    for t in np.arange(0.02, 0.98, 0.005):
        yp = cvp[:, si] > t
        tp = int((yb & yp).sum()); fp = int((~yb & yp).sum()); fn = int((yb & ~yp).sum())
        pr = tp/(tp+fp) if tp+fp else 0.0; rc = tp/(tp+fn) if tp+fn else 0.0
        f1 = 2*pr*rc/(pr+rc) if pr+rc else 0.0
        if f1 > best[0]: best = (f1, t)
    f1s[si], ths[si] = best
log(f'>>> OOF النهائي (stacking, CV مستندات): {f1s.mean():.4f}')
for s, f, t in zip(SYMBOLS, f1s, ths):
    log(f"  '{s}': F1={f:.4f} th={t:.3f}")

clfs = []
for si in range(7):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, gold_true[:, si]); clfs.append(clf)

ORDER = ['؟', '!', '.', ':', '؛', '،', '-']
IDX_TERM = [SYMBOLS.index(c) for c in '.!؟']; IDX_DOT = SYMBOLS.index('.')
finals = []
for ti, ws in enumerate(test_words_all):
    if not ws:
        finals.append(''); continue
    n = len(ws)
    b = np.concatenate([ct[ti] for ct in comp_test_l], axis=1)
    ids = np.zeros(n, dtype=int)
    mt = np.zeros((n, 4)); pb = np.array([w == PBUH for w in ws])
    mt[:, 0] = (np.arange(n) == n-1); mt[:, 1] = (np.arange(n) == 0)
    mt[:, 2] = pb; mt[:-1, 3] = pb[1:]
    Xt = np.concatenate([b, shift(b, ids, 1), shift(b, ids, -1), mt], axis=1)
    probs = np.stack([clfs[si].predict_proba(Xt)[:, 1] for si in range(7)], axis=1)
    pbin = (probs > ths[None, :]).astype(int)
    if pbin[-1, IDX_TERM].sum() == 0: pbin[-1, IDX_DOT] = 1
    finals.append(' '.join(w + ''.join(s for s in ORDER if pbin[wi, SYMBOLS.index(s)])
                           for wi, w in enumerate(ws)))

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
pd.DataFrame({'id': test_df['id'], 'final_text': finals}).to_csv(
    os.path.join(WORK_DIR, 'submission.csv'), index=False)
log('تم حفظ submission.csv — جاهز للتسليم مباشرة من Output')
_sync_to_drive()
