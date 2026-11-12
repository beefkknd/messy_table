"""Build plain-English Part Two from saved results. No inference or training.

Run with research/self-learning-program/engineering/.venv/bin/python.
The November 11 report and source are preserved under archive/2026-11-11/.
"""
from pathlib import Path
import hashlib
import html
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LATEST = ROOT/'research/detector-scale-and-integration'
SOURCES = {}


def read(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pct(value):
    return f'{100*value:.2f}%'


def table(rows, headings, table_id):
    return ('<div class="table-wrap"><table id="'+table_id+'"><thead><tr>'
            + ''.join('<th scope="col">'+h+'</th>' for h in headings)
            + '</tr></thead><tbody>'
            + ''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)
            + '</tbody></table></div>')


def figure_save(fig, name):
    for suffix in ('png', 'svg'):
        fig.savefig(HERE/'assets'/f'{name}.{suffix}', dpi=170, bbox_inches='tight')
    plt.close(fig)


def detection_chart(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0))
    colors = ['#83919c', '#4c7095', '#c19037', '#ac765c',
              '#c04b72', '#d9992b', '#147f83', '#6856b2']
    offsets = [(8,-16),(-32,10),(-27,-20),(8,-18),(-32,-22),(-36,16),(10,-15),(8,9)]
    for ax in axes:
        for i, row in enumerate(rows):
            x, y = 100*row['recall'], 100*row['precision']
            ax.scatter(x, y, s=85, color=colors[i], edgecolors='white', zorder=3)
            if ax is axes[0] and i >= 4:
                continue
            if ax is axes[1] and i < 1:
                continue
            ax.annotate(row['letter'], (x,y), xytext=offsets[i], textcoords='offset points',
                        fontsize=11, weight='bold', color=colors[i],
                        arrowprops={'arrowstyle':'-', 'color':colors[i], 'lw':.8})
        ax.set(xlabel='Objects found: recall (%) →', ylabel='Reliable boxes: precision (%) ↑')
        ax.grid(alpha=.18); ax.set_axisbelow(True)
    axes[0].scatter(100,100,marker='*',s=130,color='#233e43')
    axes[0].annotate('Ideal human-box target', (100,100),xytext=(-130,-20),textcoords='offset points',fontsize=9)
    axes[0].set(xlim=(0,104),ylim=(0,105),title='The whole journey: upper right is better')
    axes[1].set(xlim=(83,95.5),ylim=(64,91),title='A closer look at the adapted detectors')
    fig.subplots_adjust(left=.07,right=.98,top=.90,bottom=.15,wspace=.23)
    figure_save(fig,'study-detection-map')


def coverage_chart(rows):
    fig,ax=plt.subplots(figsize=(12,5.8))
    colors=['#83919c','#4c7095','#c19037','#ac765c','#c04b72','#d9992b','#147f83','#6856b2']
    offsets=[(7,8),(7,-17),(7,8),(7,-17),(12,-22),(12,-3),(12,14),(-20,15)]
    for row,color,offset in zip(rows,colors,offsets):
        c=row['counts'];x=c['unmapped'];y=100*row['coverage']
        ax.scatter(x,y,color=color,s=85,edgecolors='white',zorder=3)
        ax.annotate(row['letter'],(x,y),xytext=offset,textcoords='offset points',fontsize=11,weight='bold',color=color,
                    arrowprops={'arrowstyle':'-','color':color,'lw':.8})
    ax.set(xlabel='Unmapped prediction boxes: fewer ←',ylabel='Shared objects boxed in both views (%) ↑',
           xlim=(20000,195000),ylim=(20,100),title='Detection opportunities versus extra boxes: upper left is better')
    ax.grid(alpha=.18);ax.set_axisbelow(True);fig.tight_layout();figure_save(fig,'study-coverage-map')


def photo_example():
    example=read(HERE/'assets/study-example-detections.json')
    scene=example['scene'];cam=example['camera'];view=example['view']
    labels=read(ROOT/'data/messytable-full/MessyTable/labels/test.json')
    gt=labels['scenes'][scene]['cameras'][cam]['instances']
    image_path=ROOT/view['path']
    SOURCES[view['path']]=hashlib.sha256(image_path.read_bytes()).hexdigest()
    image=Image.open(image_path)
    panels=[('Human reference: target items',[x['pos'] for x in gt.values()],'#176d66'),
            ('D · 150-photo model + filtered tiles',[x['box_xyxy'] for x in view['detections']['old_G']],'#ac765c'),
            ('E · 600-photo model + tiles, 768px',[x['box_xyxy'] for x in view['detections']['600_768']],'#c04b72'),
            ('H · 2,400-photo model + tiles, 1024px',[x['box_xyxy'] for x in view['detections']['2400_1024']],'#6856b2')]
    fig,axes=plt.subplots(2,2,figsize=(13,8.2))
    for ax,(title,boxes,color) in zip(axes.flat,panels):
        ax.imshow(image);ax.axis('off');ax.set_title(f'{title} · {len(boxes)} boxes',fontsize=10)
        for x0,y0,x1,y1 in boxes:
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor=color,lw=1.3))
    fig.tight_layout();figure_save(fig,'study-same-photo')
    return {'scene':scene,'camera':cam,'path':view['path'],
            'selection':example['selection']}


def validate_prose(historical, one, teacher):
    """Guard prose values as well as the generated result tables."""
    inventory=read(ROOT/'research/self-learning-program/data/audit.json')['native_inventory_snapshot']
    assert inventory['train']['images']==16920
    assert sum(v['images'] for v in inventory.values())==50211
    scope=historical['scope']
    assert (scope['photos'],scope['scenes'],scope['setups'])==(16749,1861,188)
    assert pct(scope['photos']/scope['downloaded_photos'])=='33.36%'
    summary=read(LATEST/'train2400.json')['summary']
    assert (summary['photos'],summary['boxes'],summary['setups'])==(2400,58022,191)
    assert summary['includes_previous600']
    protocol=read(LATEST/'protocol.json')['training']
    assert (protocol['additional_updates'],protocol['batch'],protocol['imgsz'])==(3000,4,1024)
    training=read(LATEST/'training/sliced2400/training.json')
    assert training['completed_updates']==3000
    assert f"{training['seconds']/60:.1f}"=='24.6'
    for actual,expected in [
        (pct(one['2400_1024']['metrics']['recall']),'94.01%'),
        (pct(one['2400_1024']['metrics']['precision']),'85.89%'),
        (pct(one['2400_1024']['metrics']['both_view_coverage']),'90.12%'),
        (pct(one['600_768']['metrics']['recall']),'92.45%'),
        (pct(one['600_768']['metrics']['both_view_coverage']),'87.73%'),
        (pct(historical['arms']['vanilla']['metrics']['detection_recall']),'38.22%'),
        (pct(historical['arms']['fine_tuned']['metrics']['detection_recall']),'84.61%'),
        (pct(historical['arms']['fine_tuned_tiles']['metrics']['detection_recall']),'90.08%'),
        (pct(historical['arms']['fine_tuned_tiles']['metrics']['detection_precision']),'67.04%'),
    ]:
        assert actual==expected,(actual,expected)
    assert (one['600_768']['counts']['unmapped'],one['2400_1024']['counts']['unmapped'])==(68094,63150)
    for arm,expected in [('600_768','45.08%'),('2400_1024','51.20%')]:
        c=one[arm]['counts'];assert c['gt_lt32']==8950
        assert pct(c['tp_lt32']/c['gt_lt32'])==expected
    contrasts=teacher['cohorts']['development100']['calibrated']['contrasts']
    for arm,expected in [('S','0.18'),('L','0.14'),('O','0.09')]:
        effect=contrasts[arm+'_minus_T']['effects']['both_view_coverage']
        assert f"{100*effect['observed_delta']:.2f}"==expected
        assert effect['paired_cluster_ci95'][0]<0<effect['paired_cluster_ci95'][1]


def main():
    (HERE/'assets').mkdir(exist_ok=True)
    historical=read(ROOT/'evaluation/messytable-detection-full-test/results.json')
    one=read(LATEST/'step1-results.json')
    audit=read(LATEST/'audit.json');assert audit['status']=='passed'
    teacher=read(ROOT/'research/self-learning-program/stronger-teacher/data/independent-evaluation.json')
    quality=read(ROOT/'research/self-learning-program/stronger-teacher/data/teacher-quality.json')
    read(HERE/'metric-audit.json')
    assert pct(quality['metrics']['single_precision_iou50'])=='96.97%'
    assert pct(quality['metrics']['supplement_precision_iou50'])=='80.87%'
    assert pct(quality['metrics']['supplement_precision_iou75'])=='68.41%'
    assert quality['counts']['supplement_seeds']==554
    assert quality['counts']['frames']==240
    validate_prose(historical,one,teacher)
    rows=[]
    descriptions=[('vanilla','A','Released YOLO, no local training'),
                  ('fine_tuned','B','150 labelled photos · full frame, 768px'),
                  ('fine_tuned_tiles','C','Same 150-photo model + raw tiles'),
                  ('fine_tuned_tiles_G','D','Same 150-photo model + filtered tiles')]
    for arm,letter,name in descriptions:
        r=historical['arms'][arm];c=r['counts'];m=r['metrics']
        rows.append({'letter':letter,'name':name,'counts':c,'recall':m['detection_recall'],
                     'precision':m['detection_precision'],'coverage':m['both_view_coverage']})
    names=[('600_768','E','600 labelled photos + tiles · 768px'),
           ('600_1024','F','Same 600-photo model + tiles · 1024px'),
           ('2400_768','G','2,400 labelled photos + tiles · 768px'),
           ('2400_1024','H','Same 2,400-photo model + tiles · 1024px')]
    for arm,letter,name in names:
        r=one[arm];m=r['metrics'];rows.append({'letter':letter,'name':name,'counts':r['counts'],
                                             'recall':m['recall'],'precision':m['precision'],'coverage':m['both_view_coverage']})
    for row in rows:
        c=row['counts'];assert c['images']==16749 and c['ground_truth']==408937
        assert c['shared_objects']==1486614 and c['view_pairs']==66996
        assert c['localized']+c['unmapped']==c['detections']
        assert row['recall']==c['localized']/408937
        assert row['precision']==c['localized']/c['detections']
        assert row['coverage']==c['shared_localized_both']/1486614
    assert one['old_G']['counts']['localized']==rows[3]['counts']['localized']
    detection_chart(rows);coverage_chart(rows);example=photo_example()
    detections=table([[r['letter']+' · '+r['name'],f"{r['counts']['localized']:,}",pct(r['recall']),
                       pct(r['precision']),f"{r['counts']['unmapped']:,}",pct(r['coverage'])] for r in rows],
                     ['Configuration','Found / 408,937','Recall','Precision','Unmapped boxes','Boxed in both views'], 'detection-results')
    arms=teacher['cohorts']['development100']['calibrated'];teacher_rows=[]
    for arm,name in [('T','Continue on trusted human labels'),('S','Use single-pass automatic labels'),
                     ('L','Use automatic labels from several versions of each photo'),('O','Use complete human labels on those extra photos')]:
        r=arms['arms'][arm];c=r['counts'];effect=arms['contrasts'].get(arm+'_minus_T',{}).get('effects',{}).get('both_view_coverage')
        assert c['pairs']==100 and c['images']==200 and c['shared_objects']==2173
        delta='Reference';interval='—'
        if effect:
            delta=f"{100*effect['observed_delta']:+.2f} points"
            lo,hi=effect['paired_cluster_ci95'];interval=f'{100*lo:+.2f} to {100*hi:+.2f}'
        teacher_rows.append([name,f"{c['shared_localized_both']:,} / 2,173",pct(c['shared_localized_both']/2173),delta,interval])
    teachers=table(teacher_rows,['Short training continuation','Boxed in both views','Coverage','Change vs trusted labels','95% setup interval, points'],'self-learning-results')
    size_rows=[]
    for arm,name in [('600_768','E · 600 / 768'),('600_1024','F · 600 / 1024'),
                     ('2400_768','G · 2,400 / 768'),('2400_1024','H · 2,400 / 1024')]:
        c=one[arm]['counts'];size_rows.append([name,f"{c['tp_lt32']:,} / {c['gt_lt32']:,}",pct(c['tp_lt32']/c['gt_lt32'])])
    sizes=table(size_rows,['Configuration','Small appearances found','Small-object recall'],'small-object-results')
    extent=table([[r['letter'],f"{r['counts']['ground_truth']-r['counts']['localized']:,}",f"{r['counts']['detections']:,}",
                   pct(r['counts']['localized_at_iou75']/408937)] for r in rows],
                 ['Configuration','Missed human targets','Total predicted boxes','Recall at stricter overlap, IoU ≥ 0.75'],'extent-results')
    content=(HERE/'template.html').read_text()
    for key,value in {'DETECTION_TABLE':detections,'SELF_LEARNING_TABLE':teachers,
                      'SMALL_TABLE':sizes,'EXTENT_TABLE':extent,
                      'EXAMPLE_SCENE':html.escape(example['scene']),'EXAMPLE_CAMERA':html.escape(example['camera'])}.items():
        content=content.replace('%%'+key+'%%',value)
    assert '%%' not in content
    (HERE/'report.html').write_text(content)
    archive=HERE/'archive/2026-11-11'
    (archive/'index.html').write_text((archive/'report.html').read_text().replace('<head>','<head>\n<base href="../../">',1))
    evidence={'as_of':'2026-11-12','operation':'Presentation rewrite only; no new inference or training',
              'scope':'Part Two: YOLO Step 0 image preparation, Step 1 detection and supervised/self-label training studies only',
              'detection_photos':16749,'detection_gt_appearances':408937,
              'detection_view_pairs':66996,'detection_true_links':1486614,
              'self_learning_development_pairs':100,'self_learning_true_links':2173,
              'rows':rows,'example':example,'source_sha256':SOURCES,
              'archive':'archive/2026-11-11/index.html',
              'checks':['All 8 full-test rows preserve localization denominators and TP + unmapped accounting',
                        'Prose dataset sizes, reported headline rates, training budget and teacher effects checked against sources',
                        'Detector-to-matcher results, charts and scoring dependencies excluded; reserved for Part Three',
                        'Self-learning table is sourced separately from calibrated Development100 results',
                        'No test thresholds, model checkpoints, experiment results or Part One files changed']}
    for path in [HERE/'build.py',HERE/'template.html']:
        evidence['source_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(HERE/'evidence.json',evidence)
    print(json.dumps({'report':str(HERE/'report.html'),'status':'passed','presentation_only':True}))


if __name__=='__main__':
    main()
