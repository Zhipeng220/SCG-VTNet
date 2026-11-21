from pathlib import Path
import torch,importlib.util,json,time,platform,numpy as np
import argparse
from scg_vtnet.models import baselines as md
from scg_vtnet.workspace import RUNS

parser = argparse.ArgumentParser(description="Benchmark historical classifier heads on CPU.")
parser.add_argument("--checkpoint-dir", type=Path, default=RUNS / "historical/fair_e1_e7/seed_42")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
R = args.checkpoint_dir
torch.set_num_threads(2)
torch.manual_seed(42)
results = []
for stem in ['E2_EfficientNet-B0','E7_AC-GCAT']:
 cfg=json.loads((R/f'config_{stem}.json').read_text());d=cfg['image_feature_dim']
 if cfg['model_class']=='ImageOnlyBaseline':model=md.ImageOnlyBaseline(image_feature_dim=d,num_classes=29,dropout=cfg['dropout']);features=torch.randn(1,d)
 else:model=md.HybridAttentionModel(input_dim=11,num_classes=29,num_layers=cfg['num_layers'],num_heads=cfg['num_heads'],embed_dim=cfg['embed_dim'],dropout=cfg['dropout'],image_feature_dim=d,visual_token_count=cfg['visual_token_count'],use_cross_attn=True,use_gcn_branch=False,use_gated_fusion=True,use_dynamic_graph=False,use_keypoint_refinement=True,refine_scale=cfg['refine_scale']);features=torch.randn(1,49,d)
 model.load_state_dict(torch.load(R/'weights'/f'best_model_{stem}_21kpts.pth',map_location='cpu',weights_only=True),strict=True);model.eval();k=torch.randn(1,21,11);raw=torch.randn(1,21,3);hand=torch.ones(1)
 def forward():return model(k,features,raw_xyz=raw,hand_detected=hand) if stem.startswith('E7') else model(k,features,hand_detected=hand)
 with torch.inference_mode():
  for _ in range(20):forward()
  vals=[]
  for _ in range(100):t=time.perf_counter_ns();forward();vals.append((time.perf_counter_ns()-t)/1e6)
 results.append({'config':stem,'device':'cpu','threads':2,'batch_size':1,'warmup':20,'repeats':100,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'latency_median_ms':float(np.median(vals)),'latency_mean_ms':float(np.mean(vals)),'latency_p95_ms':float(np.percentile(vals,95)),'raw_ms':vals,'input':'synthetic pre-extracted feature tensors','excludes':['media pipe','visual backbone','image preprocessing','disk io'],'historical_reproduction':False,'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results,indent=2),encoding='utf8');print(json.dumps([{k:v for k,v in x.items() if k!='raw_ms'} for x in results],indent=2))
