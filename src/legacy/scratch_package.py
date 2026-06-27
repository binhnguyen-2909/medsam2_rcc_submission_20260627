"""(1) Đóng gói 1020 mask curated -> deliverable_dataset/ + manifest.csv
   (2) Dựng eval_boxes_550.csv cho các ảnh đủ box mọi instance."""
import json, glob, os, csv, shutil
ROOT=os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
done=set(json.load(open('labels/done.json')))
# excel mass
excel={}
if os.path.isfile('processed/excel_parsed.csv'):
    for r in csv.DictReader(open('processed/excel_parsed.csv')):
        excel[r['canon']]=r
def patient(s): return s.split('^')[0]
def canon(s): return patient(s).replace('-','')

OUT='deliverable_dataset'; os.makedirs(os.path.join(OUT,'masks'),exist_ok=True)
man=[]; eval_rows=[]; n_full=0
for s in sorted(done):
    mp=f'labels/masks/{s}.png'; pp=f'labels/prompts/{s}.json'
    if not os.path.isfile(mp): continue
    shutil.copy(mp, os.path.join(OUT,'masks',s+'.png'))
    d=json.load(open(pp)) if os.path.isfile(pp) else {}
    insts=d.get('instances',[])
    boxes=[i['box'] for i in insts if i.get('box')]
    full_box = len(insts)>0 and len(boxes)==len(insts)
    er=excel.get(canon(s),{})
    man.append({'stem':s,'patient_id':patient(s),
                'image':f'data/20241212/{s}.jpg',
                'mask':f'masks/{s}.png',
                'W':d.get('W',''),'H':d.get('H',''),
                'n_objects':d.get('n_objects',len(insts)),
                'union_area_px':d.get('union_area_px',''),
                'n_boxes_saved':len(boxes),'full_box':int(full_box),
                'boxes':';'.join(','.join(str(int(v)) for v in b) for b in boxes),
                'mass_dims_cm':er.get('mass_dims',''),
                'mass_area_cm2':er.get('mass_area_cm2','')})
    if full_box:
        n_full+=1
        for b in boxes:
            eval_rows.append([s,int(b[0]),int(b[1]),int(b[2]),int(b[3])])
with open(os.path.join(OUT,'manifest.csv'),'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(man)
with open('eval_boxes_550.csv','w',newline='') as f:
    w=csv.writer(f); w.writerow(['image','x0','y0','x1','y1']); w.writerows(eval_rows)
print(f'[ĐÓNG GÓI] {len(man)} mask -> {OUT}/masks + manifest.csv')
print(f'  ảnh nhiều-object: {sum(1 for m in man if m["n_objects"]>1)} | full_box: {n_full}')
print(f'[EVAL CSV] {n_full} ảnh đủ box, {len(eval_rows)} box-row -> eval_boxes_550.csv')
