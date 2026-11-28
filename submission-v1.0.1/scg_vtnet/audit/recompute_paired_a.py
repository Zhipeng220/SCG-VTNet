
from pathlib import Path
from argparse import ArgumentParser
import csv, random
from collections import defaultdict
parser=ArgumentParser(description="Compute paired E7 accuracy differences and percentile intervals.")
parser.add_argument("predictions",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
r=args.predictions
data=defaultdict(dict)
with r.open(encoding='utf-8-sig',newline='') as f:
    for row in csv.DictReader(f):
        k=(row['condition'],row['method'])
        rid=int(row['original_index'])
        assert rid not in data[k]
        data[k][rid]=(row['true_label'],row['prediction'],row['condition_detected'])
result=[]
rng=random.Random(20261121)
for cond in ('clean','dark_light','dark_heavy','blur_light','blur_heavy','occlude_light','occlude_heavy','xyz_jitter','zero_skeleton'):
    lhs=data[cond,'E7']
    for comp in ('E2','E5'):
        rhs=data[cond,comp]
        assert lhs.keys()==rhs.keys() and all(lhs[k][0]==rhs[k][0] for k in lhs)
        delta=[int(lhs[k][0]==lhs[k][1])-int(rhs[k][0]==rhs[k][1]) for k in sorted(lhs)]
        n=len(delta)
        plus=delta.count(1);minus=delta.count(-1)
        # Empirical percentile interval, resampling entire paired images.
        draws=sorted(100*sum(rng.choices(delta,k=n))/n for _ in range(10000))
        both_correct=sum(lhs[k][0]==lhs[k][1] and rhs[k][0]==rhs[k][1] for k in lhs)
        result.append([cond,comp,n,both_correct,plus,minus,n-both_correct-plus-minus,100*(plus-minus)/n,draws[249],draws[9749]])
with args.output.open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['condition','comparator','n','both_correct','only_E7_correct','only_comparator_correct','both_wrong','E7_minus_comparator_pp','bootstrap95_lower_pp','bootstrap95_upper_pp']);w.writerows(result)
for x in result:
    if x[0] in ('blur_heavy','zero_skeleton') and x[1]=='E5': print(x)
