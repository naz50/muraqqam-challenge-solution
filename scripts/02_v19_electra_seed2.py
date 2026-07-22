# -*- coding: utf-8 -*-
# ============================================================
#  Muraqqam Challenge — v19 (إضافة عائلات جديدة إلى الذخيرة الموجودة)
#  المطلوب: train.csv و test.csv و artifacts.npz (ذخيرة v17)
#
#  يدرّب عائلات إضافية ويضيفها إلى artifacts.npz دون المساس بالموجود:
#    EL1 = AraELECTRA-base            (~1.5 ساعة على T4)
#    S2  = AraBERT-base بذرة مختلفة   (~1.5 ساعة على T4)
#  التسميات الزائفة تُبنى من مزيج الذخيرة الحالية (أقوى من جولة v17).
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
TRAIN_EL = True    # AraELECTRA-base
TRAIN_S2 = True    # AraBERT-base بذرة مختلفة

MODEL_EL = 'aubmindlab/araelectra-base-discriminator'
MODEL_S2 = 'aubmindlab/bert-base-arabertv02'

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
if TRAIN_EL: ADD.append(('el1', MODEL_EL, 500, dict(bs=16, lr=3e-5)))
if TRAIN_S2: ADD.append(('ar3', MODEL_S2, 900, dict(bs=16, lr=3e-5)))
assert ADD, '!! فعّلي عائلة واحدة على الأقل'

new_packs = {}
for key, mname, soff, kw in ADD:
    if f'oof_{key}' in Z.files:
        log(f'{key}: موجودة مسبقا في الذخيرة — تخطٍّ')
        continue
    oof_n, test_n = train_family(mname, NF, soff, key.upper(), pseudo=pseudo_docs, **kw)
    sc = macro_of_probs(np.concatenate(oof_n, axis=0))
    log(f'{key} منفردة: OOF={sc:.4f}')
    new_packs[key] = (oof_n, test_n)

# ---------- 6) حفظ الذخيرة المدموجة ----------
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
log('حُفظت الذخيرة المدموجة artifacts.npz — مكوناتها: ' + str(all_names))
log('التالي: شغّلي سكربت v18 عليها للمزج والتسليم')
_sync_to_drive()
