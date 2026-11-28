from pathlib import Path
import pandas as pd, numpy as np,json
p=Path(__file__).resolve().parents[2]; src=p/'data/asl_alphabet/processed_train';df=pd.read_csv(src/'zerofill_landmarks_mask.csv');out={}
for seed in range(42,47):
 m=pd.read_csv(p/'manifests/asl'/f'split_manifest_seed{seed}.csv'); ix=m.original_index.to_numpy(dtype=int);v=df.iloc[ix].reset_index(drop=True);out[seed]={'rows':len(m),'unique_indices':int(len(set(ix))),'all_indices':bool(set(ix)==set(range(len(df)))),'image_mismatch':int((v.image_path!=m.image_path).sum()),'label_mismatch':int((v.label!=m.label).sum()),'detection_mismatch':int((v.hand_detected!=m.hand_detected).sum()),'counts':m.split.value_counts().to_dict(),'missing_classes_per_split':int((m.groupby(['split','label']).size().unstack(fill_value=0)==0).sum().sum())}
(p/'results'/'asl_label_mapping_lexical.json').write_text(json.dumps({k:i for i,k in enumerate(sorted(df.label.unique()))},indent=2),encoding='utf8');(p/'results'/'asl_manifest_integrity.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
