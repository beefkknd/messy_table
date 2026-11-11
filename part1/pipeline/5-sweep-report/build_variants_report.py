"""A first-stage report that starts with the matching task and fixed sample."""
import html
import json

from variant_common import OUT, SAMPLE, read, save, sha


def main():
    result = read(OUT / 'results.json')
    selection = read(SAMPLE / 'selection.json')
    payload = json.dumps(result, separators=(',', ':')).replace('<', '\\u003c')
    template = """<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Matching the same object across views · MessyTable benchmark</title>
<style>
*{box-sizing:border-box}body{margin:0;color:#1b3036;background:#fff;font:16px/1.55 Inter,system-ui,sans-serif}
main{max-width:1180px;margin:40px auto;padding:0 22px}h1{font-size:clamp(28px,4vw,43px);line-height:1.14;max-width:900px}h2{font-size:24px;line-height:1.25;margin-top:38px}h3{font-size:17px;line-height:1.4;margin:0 0 12px}
p{max-width:1000px}a{color:#146d7c}nav{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0 32px}
.table-wrap{overflow:auto;border:1px solid #d7e2e6;border-radius:7px;margin:18px 0 25px}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:11px 13px;border-bottom:1px solid #e1e9ec;white-space:nowrap}th{background:#edf3f5}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:22px 0}.card{border:1px solid #d2e3dc;border-radius:7px;background:#f1f8f4;padding:20px}.card p{margin:5px 0}.card strong{font-size:29px;line-height:1.2}
.note{background:#f0f5f7;border-left:4px solid #267b88;padding:12px 17px;margin:20px 0}.muted{color:#526c75;font-size:14px}
.flow{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:22px 0}.flow span{border:1px solid #ccdce2;padding:10px 14px;border-radius:5px}.flow b{color:#53717b}
.controls{display:flex;flex-wrap:wrap;gap:15px;margin:18px 0}label{display:flex;flex-direction:column;gap:4px;min-width:0;max-width:100%;font-weight:600}select,input{font:inherit;padding:8px;border:1px solid #aec0c8;border-radius:5px;min-width:0;max-width:100%}
details{margin:24px 0}summary{font-size:19px;font-weight:650;cursor:pointer}.reference{background:#f4f7fb}
@media(max-width:680px){main{padding:0 16px;margin:25px auto}.cards{grid-template-columns:1fr}th,td{padding:9px}.flow{font-size:14px}}
</style></head><body><main>
<h1>Which objects are the same across these photos?</h1>
<p>We want to identify the same physical item when different cameras see it from different angles, amid clutter and similar-looking objects. MessyTable gives us labeled scenes for testing this question.</p>
<p>For this stage, every method receives the same human-annotated object boxes. We compare how well the methods associate those objects across views. All models use frozen pretrained checkpoints.</p>
<div class="flow" aria-label="Experiment flow"><span>Full-scene photos + human boxes</span><b>→</b><span>Score possible object matches</span><b>→</b><span>Associate objects across views</span><b>→</b><span>Check labeled identities</span></div>
<nav aria-label="Report sections"><a href="#data">The benchmark</a><a href="#references">Reference results</a><a href="#variations">Our variations</a><a href="#top">Top three</a><a href="#scores">All results</a></nav>
<section id="data"><h2>The benchmark we kept</h2>
<p><b>100 scenes from 100 different camera setups, 900 photos and 3,600 view pairs.</b> We retain every pair among the nine cameras in each scene. The selection contains 30 easy, 50 medium and 20 hard scenes, chosen by a fixed hash rule.</p>
<p>There are 21,706 annotated object appearances and 78,335 true cross-view object pairs among 2,484,124 possible pairs. A positive match means the same physical item.</p>
<p class="muted">This is a 100-scene sample from the official test split. Our older 100-pair experiments have their own data selection and scoring settings.</p>
<p><a href="../selection.json">Preserved scene selection</a> · <a href="../RESULTS.txt">Official-protocol sample baseline</a></p></section>
<section id="references"><h2>What we are comparing against</h2>
<p>ASNet represents the original appearance-and-surroundings approach. ViT-P3DE Tiny uses a transformer with learned camera and image-position information. The published full-test numbers provide context, and the runs on our fixed sample establish the controls for every variation below.</p>
<div class="table-wrap"><table><thead><tr><th>Model</th><th>Population</th><th>Average precision ↑</th><th>False-positive rate at 95% recall ↓</th><th>Fully correct image pairs ↑</th></tr></thead><tbody id="references-body"></tbody></table></div>
<p class="muted">AP evaluates the ranking across thresholds. FPR95 measures false matches when recovering 95% of true object pairs. IPAA100 is the fraction of image pairs with every object association and absence decision correct under the authors’ assignment procedure.</p>
<p>Published figures come from the <a href="https://arxiv.org/html/2007.14878">ASNet paper</a> and <a href="https://ijcai.org/proceedings/2023/149">ViT-P3DE paper</a>. The full-test claims remain unverified by our sampled experiment.</p></section>
<section id="variations"><h2>What we changed in matching</h2>
<div class="table-wrap"><table><thead><tr><th>Direction</th><th>Question tested</th></tr></thead><tbody>
<tr><td>Object appearance and context</td><td>Do tight crops, wider context or multiple crop sizes improve DINOv2?</td></tr>
<tr><td>Foreground and image processing</td><td>Do SAM masks, GrabCut masks, grayscale or edges improve the object representation?</td></tr>
<tr><td>Orientation and camera geometry</td><td>Do quarter-turns, shared-view rotation or calibrated geometric constraints improve scores?</td></tr>
<tr><td>Perspective correction</td><td>Does rectifying a validated visible surface improve over the same unwarped surface?</td></tr>
<tr><td>Dense correspondence and fusion</td><td>Can RoMa pixel correspondence, foreground masks and fixed appearance blends improve the references?</td></tr>
</tbody></table></div>
<p>We reuse prior crop scales, rotation angles, geometry rules and fusion weights. Perspective fitting uses foreground keypoints and a held-out correspondence check before rectifying a surface patch. A native-patch control compares the same visible area without the warp.</p>
<div class="table-wrap"><table><thead><tr><th>Change tested</th><th>Control AP</th><th>Modified AP</th></tr></thead><tbody id="changes-body"></tbody></table></div>
</section>
<section id="top"><h2>Top three configurations on this sample</h2>
<p id="top-summary"></p><div class="cards" id="top-cards"></div>
<p class="muted">Ranking uses AP first, then lower FPR95, then higher IPAA100. The reference controls are excluded from the three selected variations. These winners were selected after comparing the fixed arms on this sample.</p></section>
<section id="scores"><h2>All scored configurations</h2>
<div class="controls"><label>Find a method<input id="search" type="search" placeholder="SAM, rotation, perspective or blend"></label><label>Sort results<select id="sort"><option value="AP">Average precision</option><option value="FPR95">Lower false-positive rate</option><option value="IPAA100">Fully correct image pairs</option></select></label></div>
<p class="muted" id="row-count"></p><div class="table-wrap"><table><thead><tr><th>Method</th><th>AP ↑</th><th>FPR95 ↓</th><th>IPAA100 ↑</th><th>Correct links</th><th>Wrong links</th><th>Link recall</th></tr></thead><tbody id="scores-body"></tbody></table></div>
<div class="note"><p>AP and FPR95 compare the score rankings across all candidate pairs. The final assignment counts use the authors’ global 95th-percentile distance normalization, original matching solver and fixed 0.70 cutoff. New scorers have fixed distance transforms and received no per-method cutoff fitting. Their IPAA and final assignment counts therefore also reflect score calibration.</p></div>
<details><summary>What was verified</summary><p>The sample manifest and human boxes were retained exactly. Every configuration scored the same 2,484,124 candidates and 78,335 positives. The ASNet and ViT reference metrics reproduced the preceding sample runs exactly. The original metric functions and shipped native Linux matching solver ran in a pinned Python 3.7 environment. Crop and dense-feature caches were checked against the original inference paths.</p><p>All labels are supplied by the benchmark. Predictions and fixed scoring policies were saved before the sweep’s label-based metric evaluation. The selected winners represent performance on this benchmark, with generalization outside this sample still to be evaluated.</p></details>
<p><a href="results.json">Complete results and per-scene outcomes</a> · <a href="plan.json">Frozen variation plan</a> · <a href="scores/plan.json">Exact score rules and input hashes</a></p>
</section></main><script type="application/json" id="benchmark-data">%%DATA%%</script><script>
const data=JSON.parse(document.getElementById('benchmark-data').textContent);
const get=id=>document.getElementById(id);
const pct=value=>(100*value).toFixed(2)+'%';
const fmt=value=>value.toLocaleString('en-US');
const displayLabel=label=>label.replaceAll(' reference','').replace(' / ',' + ');
const safe=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const refs=[['asnet_reference','ASNet',{AP:.524,FPR95:.209,IPAA100:.170}],['vit_reference','ViT-P3DE Tiny',{AP:.828,FPR95:.049,IPAA100:.435}]];
get('references-body').innerHTML=refs.flatMap(([key,label,published])=>[['Published full test',published],['Our fixed sample',data.methods[key]]].map(([population,row])=>'<tr><td>'+label+'</td><td>'+population+'</td><td>'+pct(row.AP)+'</td><td>'+pct(row.FPR95)+'</td><td>'+pct(row.IPAA100)+'</td></tr>')).join('');
get('top-cards').innerHTML=data.top_three_own_variants.map(key=>{
 const row=data.methods[key];
 return '<article class="card"><h3>'+safe(displayLabel(row.label))+'</h3><p><strong>'+pct(row.AP)+'</strong> AP</p><p>'+pct(row.FPR95)+' FPR95</p><p>'+pct(row.IPAA100)+' fully correct image pairs</p></article>';
}).join('');
const best=data.methods[data.top_three_own_variants[0]],vit=data.methods.vit_reference;
const difference=100*(best.AP-vit.AP);
get('top-summary').textContent=displayLabel(best.label)+' has the largest AP among our variations, '+pct(best.AP)+'. Its difference from the ViT reference is '+(difference>=0?'+':'')+difference.toFixed(2)+' percentage points.';
const changes=[
 ['DINOv2 foreground masking','dino_tight_1x','dino_sam'],
 ['Camera geometry with masked DINOv2','dino_sam','dino_sam_geometry_gate'],
 ['Perspective rectification of the same surface','dino_sam_geometry_unwarped','dino_sam_geometry_warped'],
 ['Foreground masks with RoMa','roma_box','roma_mask'],
 ['RoMa + SAM and ViT fusion','vit_reference','blend_roma_vit_reference_dense_first']
];
get('changes-body').innerHTML=changes.map(([label,control,modified])=>'<tr><td>'+label+'</td><td>'+pct(data.methods[control].AP)+'</td><td>'+pct(data.methods[modified].AP)+'</td></tr>').join('');
function render(){
 const query=get('search').value.toLowerCase().trim(),key=get('sort').value;
 const rows=Object.values(data.methods).filter(row=>!query||row.label.toLowerCase().includes(query)).sort((a,b)=>key==='FPR95'?a[key]-b[key]:b[key]-a[key]);
 get('row-count').textContent=rows.length+' configurations shown';
 get('scores-body').innerHTML=rows.map(row=>'<tr'+(row.method.endsWith('_reference')?' class="reference"':'')+'><td>'+safe(displayLabel(row.label))+'</td><td>'+pct(row.AP)+'</td><td>'+pct(row.FPR95)+'</td><td>'+pct(row.IPAA100)+'</td><td>'+fmt(row.author_assignment.correct)+'</td><td>'+fmt(row.author_assignment.wrong)+'</td><td>'+pct(row.author_assignment.recall)+'</td></tr>').join('');
}
get('search').addEventListener('input',render);get('sort').addEventListener('change',render);render();
</script></body></html>"""
    assert template.count('%%DATA%%') == 1
    (OUT / 'report.html').write_text(template.replace('%%DATA%%', payload))
    lines = ['MESSYTABLE FIXED 100-SCENE STEP 2 SWEEP\n\n',
             '100 scenes, 100 setups, 900 photos, 3,600 image pairs, 78,335 true object pairs. Human boxes. No training.\n',
             'Select top 3 by AP descending, then lower FPR95,then higher IPAA100.\n\n']
    for name in result['top_three_own_variants']:
        row = result['methods'][name]
        lines.append(f"{row['label']}\nAP {100*row['AP']:.2f}% | FPR95 {100*row['FPR95']:.2f}% | IPAA100 {100*row['IPAA100']:.2f}%\n\n")
    lines.append('The same ASNet and ViT sample controls reproduced exactly.\n')
    lines.append('Final assignment uses the author p95 distance normalization and 0.70 cutoff. New distances have no per-method calibration fitting.\n')
    lines.append('Winners were selected on this sample. Full published-test reproduction and generalization outside this sample remain unmeasured.\n')
    (OUT / 'RESULTS.txt').write_text(''.join(lines))
    save(OUT / 'report-verification-inputs.json', {'selection_sha256': sha(SAMPLE / 'selection.json'),
                                                 'results_sha256': sha(OUT / 'results.json'),
                                                 'html_sha256': sha(OUT / 'report.html')})
    print('Saved fixed-sample variation report.')


if __name__ == '__main__':
    main()
