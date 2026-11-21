import argparse
from pathlib import Path
import sys,json,time,random,math,os,platform,traceback
import numpy as np,pandas as pd,torch,joblib
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import accuracy_score,f1_score
from torch.utils.data import DataLoader,WeightedRandomSampler


def main():
    parser = argparse.ArgumentParser(description="Run strict NUS grouped experiments.")
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    data_dir = args.processed_dir
    output_dir = args.output_dir
    for input_file in (
        data_dir / "zerofill_landmarks_mask.csv",
        data_dir / "zerofill_visual_tokens_layer4_compact.npy",
        args.split_manifest,
    ):
        if not input_file.is_file():
            parser.error(f"File not found: {input_file}")
    if output_dir.exists():
        parser.error(f"Output directory already exists: {output_dir}")

    from scg_vtnet.models.scg_vtnet import HybridAttentionModel
    from scg_vtnet.datasets.multimodal import add_advanced_features,get_feature_columns,IndexedMultimodalDataset
    from scg_vtnet.training.losses import AnatomicalLoss,HierarchicalContrastiveLoss
    from scg_vtnet.training.train_loop import augment_spatial_pair,occlude_points_pair,unpack_batch
    from scg_vtnet.evaluation.anatomical_metrics import compute_anatomical_metrics

    output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2);df=pd.read_csv(data_dir/'zerofill_landmarks_mask.csv');m=pd.read_csv(args.split_manifest);ids={name:m.loc[m.split==name,'original_index'].to_numpy(dtype=int) for name in ['train','val','test']};assert sum(map(len,ids.values()))==len(df);assert all(len(set(ids[a])&set(ids[b]))==0 for a,b in [('train','val'),('train','test'),('val','test')]);assert m.groupby('group_id').split.nunique().max()==1
    dev=np.concatenate([ids['train'],ids['val']]);raw=np.zeros((len(df),21,3),dtype=np.float32);hand=np.zeros(len(df),dtype=np.float32);X=np.zeros((len(df),21,11),dtype=np.float32);y=np.zeros(len(df),dtype=np.int64);hand[dev]=df.iloc[dev].hand_detected.to_numpy(dtype=np.float32);raw[dev]=df.iloc[dev][get_feature_columns(df)].to_numpy().reshape(-1,21,3).astype(np.float32);raw[dev[hand[dev]==0]]=0;xdev=add_advanced_features(raw[dev].copy(),hand[dev],descriptor_mode='all');fitmask=np.isin(dev,ids['train']) & (hand[dev]==1);scaler=StandardScaler().fit(xdev[fitmask].reshape(int(fitmask.sum()),-1));X[dev]=scaler.transform(xdev.reshape(len(dev),-1)).reshape(-1,21,11).astype(np.float32);X[dev[hand[dev]==0]]=0;enc=LabelEncoder().fit(df.iloc[ids['train']].label);y[dev]=enc.transform(df.iloc[dev].label);features=np.load(data_dir/'zerofill_visual_tokens_layer4_compact.npy',mmap_mode='r');assert len(features)==len(df)
    joblib.dump(scaler,output_dir/'train_scaler.pkl');joblib.dump(enc,output_dir/'label_encoder.pkl');(output_dir/'class_mapping.json').write_text(json.dumps({str(c):int(i) for i,c in enumerate(enc.classes_)},indent=2),encoding='utf8')
    config={'study':'NUS grouped strict isolation, new training','split_seed':20261115,'training_seeds':[42,43,44],'perturbation_seeds':{'42':10042,'43':10043,'44':10044},'configs':{'F':{'refine':True,'anat':True},'A0':{'refine':True,'anat':False},'R0':{'refine':False,'anat':False},'G0':{'gate':False,'anat':True},'X0':{'cross_attention':False,'gate':True,'anat':True}},'epochs':10,'batch_size':32,'optimizer':'AdamW','max_lr':0.0003,'weight_decay':0.05,'scheduler':'OneCycleLR','validation_selection':'max validation accuracy, first tie','architecture':'Base HybridAttentionModel embed256 layers4 heads8, per-arm overrides in configs','anatomical_internal_weights':[1,2,.5],'anatomical_planarity_internal_factor':.5,'contrastive_weight':.1,'split_manifest':str(args.split_manifest),'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'pandas':pd.__version__,'device':'cpu','threads':torch.get_num_threads()},'warning':'Group-level split; subject/acquisition independence not established. New runs, not historical reproduction.'}
    (output_dir/'protocol_actual.json').write_text(json.dumps(config,indent=2),encoding='utf8')
    def dataset(ix):return IndexedMultimodalDataset(X[ix],raw[ix],features,ix,hand[ix],y[ix])
    trainset=dataset(ids['train']);valset=dataset(ids['val'])
    def make_model(name):return HybridAttentionModel(input_dim=11,num_classes=len(enc.classes_),embed_dim=256,num_layers=4,num_heads=8,dropout=.2,image_feature_dim=2048,visual_token_count=49,use_cross_attn=(name!='X0'),use_gated_fusion=(name!='G0'),use_gcn_branch=False,use_dynamic_graph=False,use_keypoint_refinement=(name!='R0'),refine_scale=.05)
    def va(model):
     model.eval();allp=[];allt=[]
     with torch.no_grad():
      for batch in DataLoader(valset,batch_size=32,shuffle=False):
       k,r,v,h,t=unpack_batch(batch,'cpu');z=model(k,v,raw_xyz=r,hand_detected=h)[0];allp.extend(z.argmax(1).tolist());allt.extend(t.tolist())
     return float(accuracy_score(allt,allp))
    for seed in [42,43,44]:
     for name,use_anat in [('F',True),('A0',False),('R0',False),('G0',True),('X0',True)]:
      run=output_dir/f'{name}_seed{seed}';run.mkdir();random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);model=make_model(name);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05);freq=np.bincount(y[ids['train']],minlength=len(enc.classes_));weights=1/(freq[y[ids['train']]]+1e-8);sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=torch.Generator().manual_seed(seed));loader=DataLoader(trainset,batch_size=32,sampler=sampler,num_workers=0);scheduler=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=.0003,epochs=10,steps_per_epoch=len(loader));ce=torch.nn.CrossEntropyLoss(label_smoothing=.1);contrast=HierarchicalContrastiveLoss(.1);anat=AnatomicalLoss();best=-1;history=[];start=time.time()
      for epoch in range(10):
       model.train();ls=[];aw=.001*.5*(1-math.cos(math.pi*epoch/9)) if use_anat else 0.
       for batch in loader:
        k,r,v,h,t=unpack_batch(batch,'cpu');k=k.clone();r=r.clone();v=v.clone();mask=h.reshape(-1,1,1)
        if torch.rand(1).item()<.8:
         if torch.rand(1).item()<.5:k,r=augment_spatial_pair(k,r)
         elif torch.rand(1).item()<.7:k=k+torch.randn_like(k)*.5;r=r+torch.randn_like(r)*.005
         else:k,r=occlude_points_pair(k,r,num_occlusions=3)
         k=k*mask;r=r*mask;v=v+torch.randn_like(v)*.1
        opt.zero_grad(set_to_none=True);z,emb,_,xyz=model(k,v,raw_xyz=r,hand_detected=h);loss=ce(z,t)+.1*contrast(emb,t)
        if aw:loss=loss+aw*anat(xyz,t,confidences=h[:,None].expand(-1,21))
        if not bool(torch.isfinite(loss)):raise RuntimeError(f'Nonfinite loss {name} {seed} {epoch}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();scheduler.step();ls.append(float(loss.detach()))
       val=va(model);record={'epoch':epoch+1,'train_loss':float(np.mean(ls)),'val_accuracy':val,'anat_weight':aw,'elapsed_s':round(time.time()-start,1)};history.append(record);(run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8');print(name,seed,record,flush=True)
       if val>best:best=val;torch.save(model.state_dict(),run/'best.pt');(run/'best.json').write_text(json.dumps({'epoch':epoch+1,'val_accuracy':val},indent=2),encoding='utf8')
    # Only after all four runs completed: hold-out test evaluation, one pre-registered pass per model and condition.
    (output_dir/'training_complete.json').write_text(json.dumps({'completed':True,'runs':15}),encoding='utf8')
    tid=ids['test'];hand[tid]=df.iloc[tid].hand_detected.to_numpy(dtype=np.float32);raw[tid]=df.iloc[tid][get_feature_columns(df)].to_numpy().reshape(-1,21,3).astype(np.float32);raw[tid[hand[tid]==0]]=0;xtest=add_advanced_features(raw[tid].copy(),hand[tid],descriptor_mode='all');X[tid]=scaler.transform(xtest.reshape(len(tid),-1)).reshape(-1,21,11).astype(np.float32);X[tid[hand[tid]==0]]=0;y[tid]=enc.transform(df.iloc[tid].label);testset=dataset(tid);from scg_vtnet.evaluation.anatomical_metrics import compute_anatomical_metrics
    metrics=[]
    for seed in [42,43,44]:
     for name in ['F','A0','R0','G0','X0']:
      run=output_dir/f'{name}_seed{seed}';model=make_model(name);model.load_state_dict(torch.load(run/'best.pt',map_location='cpu',weights_only=True),strict=True);model.eval()
      for scenario in ['clean','skeleton_gaussian','visual_token_dropout']:
       rows=[];raws=[];refs=[];hands=[];offset=0
       with torch.no_grad():
        for batch in DataLoader(testset,batch_size=32,shuffle=False):
         k,r,v,h,t=unpack_batch(batch,'cpu');idx=ids['test'][offset:offset+len(t)];offset+=len(t);k=k.clone();r=r.clone();v=v.clone()
         for j,orig in enumerate(idx):
          gen=torch.Generator().manual_seed(10000+seed+int(orig)*17+(0 if scenario=='skeleton_gaussian' else 1))
          if scenario=='skeleton_gaussian':k[j]+=torch.randn(k[j].shape,generator=gen)*1.;r[j]+=torch.randn(r[j].shape,generator=gen)*.01
          elif scenario=='visual_token_dropout':v[j]*=(torch.rand((49,1),generator=gen)>.25)
         k=k*h[:,None,None];r=r*h[:,None,None];z,_,_,ref=model(k,v,raw_xyz=r,hand_detected=h);pr=torch.softmax(z,dim=1);pred=z.argmax(1);raws.append(r);
         if ref is not None:refs.append(ref)
         hands.append(h)
         for j,i in enumerate(idx):rows.append({'original_index':int(i),'image_path':df.iloc[i].image_path,'true_label':str(enc.classes_[int(t[j])]),'pred_label':str(enc.classes_[int(pred[j])]),'pred_probability':float(pr[j,pred[j]]),'hand_detected':int(h[j]),'scenario':scenario,'training_seed':seed,'perturbation_seed':10000+seed})
       result=pd.DataFrame(rows);result.to_csv(run/f'predictions_{scenario}.csv',index=False,encoding='utf8');metric={'config':name,'training_seed':seed,'scenario':scenario,'n':len(result),'accuracy':accuracy_score(result.true_label,result.pred_label),'macro_f1':f1_score(result.true_label,result.pred_label,average='macro'),'raw_diagnostic':compute_anatomical_metrics(torch.cat(raws),torch.cat(hands)),'refined_diagnostic':compute_anatomical_metrics(torch.cat(refs),torch.cat(hands)) if refs else None};metrics.append(metric);print('TEST',json.dumps(metric),flush=True)
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8');(output_dir/'COMPLETE.json').write_text(json.dumps({'status':'complete','models':15,'conditions':3}),encoding='utf8')


if __name__ == "__main__":
    main()
