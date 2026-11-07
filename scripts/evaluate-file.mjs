import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {normalizeRecords,validateChecklist,evaluateRules,evaluateWithModel,scoreResults,parseExtraBody} from '../src/core.mjs';
const {values}=parseArgs({options:{data:{type:'string'},checklist:{type:'string'},out:{type:'string',default:'results.json'},engine:{type:'string',default:'model'},verification:{type:'string',default:'verified'},ids:{type:'string'}}});
if(!values.data||!values.checklist){console.error('Usage: node --env-file-if-exists=.env scripts/evaluate-file.mjs --data control.json --checklist official.json --out results.json [--engine rules] [--verification single] [--ids ids.txt]');process.exit(1);}
const json=async path=>JSON.parse(await readFile(resolve(path),'utf8'));
let records=normalizeRecords(await json(values.data));
// Optional subset (one conversation_id per line), e.g. the held-out evaluation split.
if(values.ids){const keep=new Set((await readFile(resolve(values.ids),'utf8')).split(/\r?\n/).map(s=>s.trim()).filter(Boolean));records=records.filter(r=>keep.has(r.conversation_id));}
const checklist=validateChecklist(await json(values.checklist));
if(!['model','rules'].includes(values.engine))throw new Error('engine must be model or rules');
if(values.engine==='model'&&(!process.env.MODEL_API_KEY||!process.env.MODEL_NAME))throw new Error('Set MODEL_API_KEY and MODEL_NAME. There is no silent fallback.');
const extraBody=parseExtraBody(process.env.MODEL_EXTRA_BODY);
const timeoutMs=Math.max(30,Math.min(300,Number(process.env.MODEL_TIMEOUT_SECONDS)||180))*1000;
const state={schemaVersion:1,records,checklists:[checklist],evaluations:[],appeals:[],createdAt:new Date().toISOString(),errors:[]};
const out=resolve(values.out);await mkdir(dirname(out),{recursive:true});
for(const record of records){
  const started=Date.now();
  try{
    const result=values.engine==='rules'?{results:evaluateRules(record,checklist),usage:null,verification:null}:await evaluateWithModel(record,checklist,{key:process.env.MODEL_API_KEY,model:process.env.MODEL_NAME,baseURL:process.env.MODEL_BASE_URL||'https://api.openai.com/v1',allowLocal:process.env.ALLOW_LOCAL_MODEL==='1',verificationMode:values.verification,timeoutMs,extraBody,inputPrice:Number(process.env.MODEL_INPUT_PRICE)||0,outputPrice:Number(process.env.MODEL_OUTPUT_PRICE)||0});
    const e={id:crypto.randomUUID(),conversationId:record.conversation_id,engine:values.engine,record,checklist,createdAt:new Date().toISOString(),durationMs:Date.now()-started,...result};
    e.originalScore=scoreResults(checklist,e.results);e.finalScore=e.originalScore;
    state.evaluations.push(e);console.log(`${record.conversation_id}: ${e.originalScore.total??'pending'}/100; calls=${e.usage?.calls||0}; ${((Date.now()-started)/1000).toFixed(1)}s`);
  }catch(error){state.errors.push({conversationId:record.conversation_id,error:error.message});console.error(`${record.conversation_id}: ${error.message}`);}
  await writeFile(out,JSON.stringify(state,null,2)+'\n');
}
console.log(`Saved ${state.evaluations.length}/${records.length} evaluations to ${out}`);
if(state.errors.length)process.exitCode=1;
