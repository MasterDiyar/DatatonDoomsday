// Turn teacher-model evaluations (optionally corrected by reviewers through appeals in the UI)
// into a label file. The file is compatible with scripts/benchmark.mjs (Ground Truth format)
// and additionally carries full results + transcript snapshots for scripts/build-sft-dataset.mjs.
//
// These are *silver* labels produced by our own team, not the organisers' Ground Truth.
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {finalResults} from '../src/core.mjs';

const {values}=parseArgs({options:{
  state:{type:'string',multiple:true},out:{type:'string',default:'datasets/labels-dev.json'},
  engine:{type:'string',default:'model'},checklist:{type:'string',default:'freedom-chat-quality'},
  'only-reviewed':{type:'boolean',default:false},'reviewed-ids':{type:'string'}
}});
if(!values.state?.length){console.error('Usage: node scripts/export-labels.mjs --state cqa-export.json [--state more.json] [--only-reviewed] [--reviewed-ids ids.txt] [--out datasets/labels-dev.json]');process.exit(1);}
const confirmed=values['reviewed-ids']?new Set((await readFile(resolve(values['reviewed-ids']),'utf8')).split(/\r?\n/).map(s=>s.trim()).filter(Boolean)):new Set();
const byId=new Map();let checklistVersion=null,skipped=0;
for(const path of values.state){
  const state=JSON.parse(await readFile(resolve(path),'utf8'));
  state.appeals??=[];
  for(const e of state.evaluations||[]){
    if(e.engine!==values.engine||e.checklist?.id!==values.checklist)continue;
    if(checklistVersion!==null&&e.checklist.version!==checklistVersion){console.warn(`${e.conversationId}: checklist version ${e.checklist.version} differs from ${checklistVersion}; skipped.`);skipped++;continue;}
    checklistVersion=e.checklist.version;
    const results=finalResults(state,e);
    const appealed=state.appeals.some(a=>a.evaluationId===e.id&&a.status==='resolved');
    const reviewed=appealed||confirmed.has(e.conversationId);
    if(values['only-reviewed']&&!reviewed)continue;
    if(results.some(r=>r.answer==='pending')){console.warn(`${e.conversationId}: contains pending answers; resolve them in the UI first. Skipped.`);skipped++;continue;}
    // Later files / later evaluations win.
    byId.set(e.conversationId,{conversationId:e.conversationId,language:e.record.language,channel:e.record.channel,reviewed,
      answers:Object.fromEntries(results.map(r=>[r.criterionId,r.answer])),
      results:results.map(({criterionId,answer,explanation,evidence,contextEvidence,checkedRange})=>({criterionId,answer,explanation,evidence:evidence||[],contextEvidence:contextEvidence||[],checkedRange:checkedRange||null})),
      record:e.record});
  }
}
if(!byId.size){console.error('No matching evaluations found.');process.exit(1);}
const file={checklistId:values.checklist,checklistVersion,engine:values.engine,labelKind:'silver: teacher model + team review',createdAt:new Date().toISOString(),records:[...byId.values()]};
const out=resolve(values.out);await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify(file,null,2)+'\n');
console.log(`Wrote ${file.records.length} labels (${file.records.filter(r=>r.reviewed).length} reviewed, ${skipped} skipped) to ${out}`);
