// Export evaluations to results/<dev|prod>/results.json (organisers' JSON format 1.0).
// Input is either the state file of scripts/evaluate-file.mjs or the full UI export (/api/export).
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {buildSubmission,validateSubmission,datasetIds} from '../src/hackathon-export.mjs';

const {values}=parseArgs({options:{
  state:{type:'string'},data:{type:'string'},dataset:{type:'string'},out:{type:'string'},
  team:{type:'string',default:process.env.TEAM_NAME||''},
  'model-name':{type:'string',default:process.env.SUBMISSION_MODEL_NAME||process.env.MODEL_NAME||''},
  mode:{type:'string',default:process.env.SUBMISSION_MODEL_MODE||'local'},
  engine:{type:'string',default:'model'},checklist:{type:'string',default:'freedom-chat-quality'},
  'use-appeals':{type:'boolean',default:false},expect:{type:'string'},'validate-only':{type:'string'}
}});
const json=async path=>JSON.parse(await readFile(resolve(path),'utf8'));
const expectCount=values.expect!==undefined?Number(values.expect):values.dataset==='prod'?10:null;

if(values['validate-only']){
  const file=await json(values['validate-only']);
  const expectIds=values.data?datasetIds(await json(values.data)):null;
  const problems=validateSubmission(file,{expectIds,expectCount});
  if(problems.length){console.error(problems.join('\n'));process.exit(1);}
  console.log(`OK: ${file.results.length} entries, ${file.results.filter(r=>r.status==='error').length} errors.`);
  process.exit(0);
}
if(!values.state||!['dev','prod'].includes(values.dataset)){
  console.error('Usage: node scripts/export-hackathon.mjs --state eval-state.json --dataset dev|prod [--data dataset.json] --team "Team" --model-name "Model" [--mode local|external_api|hybrid] [--engine model|rules] [--use-appeals] [--out results/dev/results.json]\n       node scripts/export-hackathon.mjs --validate-only results/prod/results.json [--data dataset.json] [--expect 10]');
  process.exit(1);
}
const state=await json(values.state);
const ids=values.data?datasetIds(await json(values.data)):null;
const file=buildSubmission({state,ids,teamName:values.team,modelName:values['model-name'],mode:values.mode,engine:values.engine,checklistId:values.checklist,useAppeals:values['use-appeals']});
const problems=validateSubmission(file,{expectIds:ids,expectCount});
const out=resolve(values.out||`results/${values.dataset}/results.json`);
await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify(file,null,2)+'\n','utf8');
const errors=file.results.filter(r=>r.status==='error');
console.log(`Wrote ${file.results.length} entries (${errors.length} errors) to ${out}`);
for(const e of errors)console.warn(`  error ${e.conversation_id}: ${e.error}`);
if(problems.length){console.error('Validation problems:\n'+problems.join('\n'));process.exitCode=1;}
