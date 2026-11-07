// Build chat-format SFT data for fine-tuning a local model.
// Prompts come from buildModelMessages() in src/core.mjs, i.e. exactly what the evaluator sends
// at inference time; the assistant target is the validated {"results":[...]} JSON.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import {parseArgs} from 'node:util';
import {createHash} from 'node:crypto';
import {validateChecklist,buildModelMessages,validateModelResults} from '../src/core.mjs';

const {values}=parseArgs({options:{
  labels:{type:'string',multiple:true},checklist:{type:'string',default:'examples/checklist-official.json'},
  'out-dir':{type:'string',default:'datasets/sft'},'holdout-ids':{type:'string'},
  'holdout-fraction':{type:'string',default:'0.2'},'max-chars':{type:'string',default:'60000'},
  scenarios:{type:'string'},'synthetic-prefix':{type:'string',default:'synth-'}
}});
if(!values.labels?.length){console.error('Usage: node scripts/build-sft-dataset.mjs --labels datasets/labels-dev.json [--labels datasets/labels-synth.json] [--scenarios datasets/synth-scenarios.json] [--holdout-ids ids.txt | --holdout-fraction 0.2] [--out-dir datasets/sft]');process.exit(1);}
const json=async path=>JSON.parse(await readFile(resolve(path),'utf8'));
const checklist=validateChecklist(await json(values.checklist));
const maxChars=Number(values['max-chars']);
const scenarios=values.scenarios?await json(values.scenarios):{};
const synthetic=id=>id.startsWith(values['synthetic-prefix']);
// Deterministic split: same IDs always land in the same split.
const fraction=Number(values['holdout-fraction']);
const holdout=values['holdout-ids']?new Set((await readFile(resolve(values['holdout-ids']),'utf8')).split(/\r?\n/).map(s=>s.trim()).filter(Boolean)):null;
const inHoldout=id=>!synthetic(id)&&(holdout?holdout.has(id):parseInt(createHash('sha256').update(id).digest('hex').slice(0,8),16)/0xffffffff<fraction);

const train=[],evalSet=[],truthRecords=[],stats={kept:0,skipped:0,tooLong:0,contradicted:0,invalid:0,byLanguage:{}};
let labelMeta=null;
for(const path of values.labels){
  const file=await json(path);
  if(file.checklistId!==checklist.id){console.warn(`${path}: checklist ${file.checklistId} ≠ ${checklist.id}; skipped.`);continue;}
  labelMeta??={checklistId:file.checklistId,checklistVersion:file.checklistVersion,engine:file.engine||'model'};
  for(const label of file.records){
    const id=label.conversationId,record=label.record;
    if(!record?.transcript){console.warn(`${id}: label has no transcript snapshot; skipped.`);stats.skipped++;continue;}
    // Synthetic dialogues were generated with intended outcomes; drop those the teacher contradicts.
    const intent=scenarios[id]?.expected||{};
    const conflicts=Object.entries(intent).filter(([k,v])=>Object.hasOwn(label.answers,k)&&JSON.stringify(label.answers[k])!==JSON.stringify(v));
    if(conflicts.length){console.warn(`${id}: teacher label contradicts scenario (${conflicts.map(([k])=>k).join(', ')}); skipped.`);stats.contradicted++;continue;}
    const target={results:checklist.criteria.map(c=>{
      const r=label.results.find(r=>r.criterionId===c.id);
      if(!r)return null;
      const out={criterionId:c.id,answer:r.answer,explanation:r.explanation,evidence:r.evidence||[],contextEvidence:r.contextEvidence||[]};
      if(r.checkedRange)out.checkedRange=r.checkedRange;
      return out;
    })};
    if(target.results.includes(null)){console.warn(`${id}: missing criteria in label; skipped.`);stats.invalid++;continue;}
    try{validateModelResults(target,record,checklist);}catch(error){console.warn(`${id}: label fails evaluator validation (${error.message}); fix it in the UI. Skipped.`);stats.invalid++;continue;}
    const messages=[...buildModelMessages(record,checklist),{role:'assistant',content:JSON.stringify(target)}];
    const chars=messages.reduce((s,m)=>s+m.content.length,0);
    if(chars>maxChars){console.warn(`${id}: ${chars} chars > --max-chars ${maxChars}; skipped.`);stats.tooLong++;continue;}
    const row={conversation_id:id,language:record.language,synthetic:synthetic(id),chars,messages};
    if(inHoldout(id)){evalSet.push(row);truthRecords.push({conversationId:id,answers:label.answers});}else train.push(row);
    stats.kept++;stats.byLanguage[record.language]=(stats.byLanguage[record.language]||0)+1;
  }
}
const outDir=resolve(values['out-dir']);await mkdir(outDir,{recursive:true});
const jsonl=rows=>rows.map(r=>JSON.stringify(r)).join('\n')+(rows.length?'\n':'');
await writeFile(join(outDir,'train.jsonl'),jsonl(train));
await writeFile(join(outDir,'eval.jsonl'),jsonl(evalSet));
await writeFile(join(outDir,'holdout-ids.txt'),evalSet.map(r=>r.conversation_id).join('\n')+'\n');
if(labelMeta)await writeFile(join(outDir,'holdout-truth.json'),JSON.stringify({...labelMeta,labelKind:'silver',records:truthRecords},null,2)+'\n');
const lengths=[...train,...evalSet].map(r=>r.chars).sort((a,b)=>a-b);
const pct=p=>lengths.length?lengths[Math.min(lengths.length-1,Math.floor(p*lengths.length))]:0;
console.log(`train=${train.length} eval=${evalSet.length} skipped: tooLong=${stats.tooLong} invalid=${stats.invalid} contradicted=${stats.contradicted} other=${stats.skipped}`);
console.log(`by language: ${JSON.stringify(stats.byLanguage)}; chars p50=${pct(0.5)} p95=${pct(0.95)} max=${lengths.at(-1)||0} (~${Math.ceil((lengths.at(-1)||0)/3)} tokens)`);
console.log(`Wrote ${outDir}/{train.jsonl,eval.jsonl,holdout-ids.txt,holdout-truth.json}`);
