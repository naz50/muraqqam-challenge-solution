# -*- coding: utf-8 -*-
# ============================================================
#  Muraqqam Challenge — v28 (تصويت الأغلبية بين ثلاثة أنظمة — CPU فقط، دقيقة واحدة)
#  لا يحتاج GPU إطلاقا — شغليه في نوتبوك Kaggle عادي (Accelerator: None).
#
#  المدخلات (ارفعيها كـ Dataset واحد):
#    - submission_old.csv   (نظامك القديم — 0.688)
#    - submission_v25.csv   (مخرج v25 — 0.681)
#    - submission_v26.csv   (مخرج v26 — 0.685)
#  + بيانات المسابقة (test.csv) من مدخل المسابقة نفسها.
#
#  الطريقة: لكل موضع كلمة، تُعتمد العلامة إذا اتفق عليها نظامان من ثلاثة.
#  + قاعدة النهاية (آخر كلمة تحتاج علامة إنهاء) + فحص المحاذاة الرسمي.
# ============================================================

import os, glob
import pandas as pd

WORK_DIR = ('/kaggle/working' if os.path.exists('/kaggle/working') else '.')
VALID = set('.،؟!:؛-')
SYMBOLS = sorted(VALID)
ORDER = ['؟', '!', '.', ':', '؛', '،', '-']
TERM = set('.!؟')

def tok(pred):
    """مُقسِّم المقياس الرسمي: (كلمة، رموز الفجوة بعدها)."""
    pairs, cur, in_word = [], [], False
    for ch in str(pred):
        if ch.isspace():
            if in_word: pairs.append([''.join(cur), []]); cur=[]; in_word=False
            continue
        if ch in VALID:
            if in_word: pairs.append([''.join(cur), [ch]]); cur=[]; in_word=False
            else:
                if pairs: pairs[-1][1].append(ch)
            continue
        if not in_word: in_word=True; cur=[ch]
        else: cur.append(ch)
    if in_word: pairs.append([''.join(cur), []])
    return pairs

def find(pattern):
    hits = []
    for root in ('/kaggle/input', '.'):
        if os.path.exists(root):
            hits += glob.glob(os.path.join(root, '**', pattern), recursive=True)
    return sorted(set(hits), key=lambda p: (len(p), p))

test_path = (find('test.csv') or [None])[0]
srcs = []
SYSTEMS = [
    ('القديم 0.688', ['submission_old*.csv', 'submission (18)*.csv', 'submission*18*.csv']),
    ('v25',          ['submission_v25*.csv', 'submission v25*.csv', '*v25*.csv']),
    ('v26',          ['submission_v26*.csv', '*v26*.csv']),
]
for name, pats in SYSTEMS:
    p = None
    for pat in pats:
        hits = [h for h in find(pat) if 'sample_submission' not in h]
        if hits: p = hits[0]; break
    assert p, f'!! لم أجد ملف نظام {name} — الأنماط المجربة: {pats}'
    srcs.append(p)
assert test_path, '!! أضيفي بيانات المسابقة'
print('test:', test_path)
for p in srcs: print('نظام:', p)

test_df = pd.read_csv(test_path)
systems = []
for p in srcs:
    df = pd.read_csv(p)
    assert len(df) == len(test_df) and (df['id'].values == test_df['id'].values).all(), f'{p}: صفوف/معرفات لا تطابق'
    sets_per_doc = []
    for raw, pred in zip(test_df['text'], df['final_text']):
        pairs = tok(pred)
        assert [w for w, _ in pairs] == str(raw).strip().split(), f'{p}: محاذاة خاطئة'
        sets_per_doc.append([set(c for c in g if c in VALID) for _, g in pairs])
    systems.append(sets_per_doc)
print('حُمّلت الأنظمة الثلاثة وتحققت محاذاتها ✓')

finals, changed = [], [0, 0, 0]
for ti in range(len(test_df)):
    ws = str(test_df['text'].iloc[ti]).strip().split()
    parts = []
    for wi in range(len(ws)):
        votes = [systems[k][ti][wi] for k in range(3)]
        chosen = set(s for s in SYMBOLS if sum(s in v for v in votes) >= 2)
        if wi == len(ws) - 1 and not (chosen & TERM):
            chosen.add('.')
        for k in range(3):
            changed[k] += chosen != votes[k]
        parts.append(ws[wi] + ''.join(s for s in ORDER if s in chosen))
    finals.append(' '.join(parts))

for raw, fin in zip(test_df['text'], finals):
    assert [w for w, _ in tok(fin)] == str(raw).strip().split(), 'word mismatch!'
print(f'التغييرات عن كل نظام: old={changed[0]} | v25={changed[1]} | v26={changed[2]}')
print(f'فحص المحاذاة: كل الصفوف الـ {len(finals)} صالحة')

out = os.path.join(WORK_DIR, 'submission.csv')
pd.DataFrame({'id': test_df['id'], 'final_text': finals}).to_csv(out, index=False)
print('تم حفظ', out)
