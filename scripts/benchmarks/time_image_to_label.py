from pathlib import Path
import argparse,sys,time,json,platform,importlib.util
import numpy as np,pandas as pd,torch,torchvision,mediapipe as mp,cv2,joblib
from PIL import Image
from torchvision import models,transforms
from torchvision.models.feature_extraction import create_feature_extractor



from scg_vtnet.models.baselines import HybridAttentionModel, ImageOnlyBaseline
from scg_vtnet.datasets import visual_features as spatial, multimodal as fu
from scg_vtnet.workspace import ASL_IMAGES, ASL_SPLITS, RUNS

parser=argparse.ArgumentParser(description="Time E7 and E2 on one fixed image per class.")
parser.add_argument("--split", choices=("train","test"), default="train")
parser.add_argument("--output", type=Path, required=True)
args=parser.parse_args()
R=RUNS / "historical/fair_e1_e7/seed_42"
root=ASL_IMAGES
manifest=pd.read_csv(ASL_SPLITS / "split_manifest_seed42.csv")
sm=manifest[manifest.split==args.split].sort_values(["label","image_path"]).groupby("label",sort=True).head(1)
assert len(sm)==sm.label.nunique()==29
images=[str(root/Path(str(x).replace(chr(92),"/"))) for x in sm.image_path]
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2);device='cpu';cfg=json.loads((R/'config_E7_AC-GCAT.json').read_text());model=HybridAttentionModel(input_dim=11,num_classes=29,num_layers=4,num_heads=8,embed_dim=256,dropout=.2,image_feature_dim=2048,visual_token_count=49,use_cross_attn=True,use_gcn_branch=False,use_gated_fusion=True,use_dynamic_graph=False,use_keypoint_refinement=True,refine_scale=.05);model.load_state_dict(torch.load(R/'weights'/'best_model_E7_AC-GCAT_21kpts.pth',map_location='cpu',weights_only=True),strict=True);model.eval();scaler=joblib.load(R/'scaler_E7_AC-GCAT.pkl');
back=spatial.build_resnet50_extractor('spatial_tokens',torch.device('cpu'));prep=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])]);det=mp.solutions.hands.Hands(static_image_mode=True,max_num_hands=1,min_detection_confidence=.3)
ecfg=json.loads((R/'config_E2_EfficientNet-B0.json').read_text());em=ImageOnlyBaseline(image_feature_dim=1280,num_classes=29,dropout=.2);em.load_state_dict(torch.load(R/'weights'/'best_model_E2_EfficientNet-B0_21kpts.pth',map_location='cpu',weights_only=True),strict=True);em.eval();weights=models.EfficientNet_B0_Weights.DEFAULT;eb=models.efficientnet_b0(weights=weights);eb.classifier=torch.nn.Identity();eb.eval();et=weights.transforms();dummy_k=torch.zeros((1,21,11));dummy_hand=torch.ones(1)
def ms(a,b):return (b-a)/1e6
def e7(path):
 t=time.perf_counter_ns();img=spatial.canonical_rgb_image(path,224,'letterbox');a=time.perf_counter_ns();rr=det.process(img);h=int(bool(rr.multi_hand_landmarks));b=time.perf_counter_ns();tt=prep(Image.fromarray(img)).unsqueeze(0)
 with torch.inference_mode(): v=back(tt)['feat'].flatten(2).transpose(1,2)
 c=time.perf_counter_ns();raw=np.array(spatial.extract_21x3_landmarks(rr) if h else [0.]*63,dtype=np.float32).reshape(1,21,3);feat=fu.add_advanced_features(raw.copy(),np.array([h],dtype=np.float32),'all');scaled=scaler.transform(feat.reshape(1,-1)).reshape(1,21,11).astype(np.float32) if h else np.zeros((1,21,11),dtype=np.float32);k=torch.from_numpy(scaled);rk=torch.from_numpy(raw);hh=torch.tensor([h],dtype=torch.float32);d=time.perf_counter_ns()
 with torch.inference_mode():out=model(k,v,raw_xyz=rk,hand_detected=hh)[0];pred=int(out.argmax(1))
 e=time.perf_counter_ns();return {'decode_resize_ms':ms(t,a),'detection_ms':ms(a,b),'visual_preprocess_backbone_ms':ms(b,c),'aar_scale_ms':ms(c,d),'head_ms':ms(d,e),'total_ms':ms(t,e),'hand_detected':h,'prediction_class_index':pred}
def e2(path):
 t=time.perf_counter_ns()
 with Image.open(path) as img:tt=et(img.convert('RGB')).unsqueeze(0)
 a=time.perf_counter_ns()
 with torch.inference_mode():v=eb(tt)
 b=time.perf_counter_ns()
 with torch.inference_mode():out=em(dummy_k,v,hand_detected=dummy_hand)[0];pred=int(out.argmax(1))
 c=time.perf_counter_ns();return {'decode_transform_ms':ms(t,a),'backbone_ms':ms(a,b),'head_ms':ms(b,c),'total_ms':ms(t,c),'prediction_class_index':pred}
for fn in [e7,e2]:
 for path in images[:5]:fn(path)
results={}
for name,fn in [('E7',e7),('E2',e2)]:
 data=[dict(path=path,**fn(path)) for path in images];cols=[k for k in data[0] if k.endswith('_ms')];summary={k:{'median':float(np.median([r[k] for r in data])),'p95':float(np.percentile([r[k] for r in data],95))} for k in cols};results[name]={'rows':data,'summary':summary}
det.close();report={'status':'complete','scope':'CPU image-to-label timing; no accuracy estimate','split':args.split,'split_seed':42,'selection':'first image per class by sorted manifest path','threads':2,'batch_size':1,'warmup_each':5,'measured_images':29,'hardware':platform.processor(),'os':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'mediapipe':mp.__version__,'cv2':cv2.__version__,'results':results};output.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps({k:v['summary'] for k,v in results.items()},indent=2))
