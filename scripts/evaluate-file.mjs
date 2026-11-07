import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {normalizeRecords,validateChecklist,evaluateRules,evaluateWithModel,scoreResults,parseExtraBody} from '../src/core.mjs';
import {rawRecords} from '../src/hackathon-export.mjs';
const {values}=parseArgs({options:{data:{type:'string'},checklist:{type:'string'},out:{type:'string',default:'results.json'},engine:{type:'string',default:'model'},verification:{type:'string',default:'verified'},ids:{type:'string'},retries:{type:'string',default:'1'}}});
if(!values.data||!values.checklist){console.error('Usage: node --env-file-if-exists=.env scripts/evaluate-file.mjs --data control.json --checklist official.json --out results.json [--engine rules] [--verification single] [--ids ids.txt] [--retries 1]');process.exit(1);}
const json=async path=>JSON.parse(await readFile(resolve(path),'utf8'));
const payload=await json(values.data);
if(payload?.isSuccess===false)throw new Error('The dataset source reported a loading error.');
// Validate each record separately: one malformed conversation must become an error entry
// with its original ID, not abort the whole dataset.
const records=[],errors=[],seen=new Set();
let raw=rawRecords(payload);
if(values.ids){const keep=new Set((await readFile(resolve(values.ids),'utf8')).split(/\r?\n/).map(s=>s.trim()).filter(Boolean));raw=raw.filter(r=>keep.has(String(r?.conversation_id??'').trim()));}
for(const [i,r] of raw.entries()){
  const id=typeof r?.conversation_id==='string'&&r.conversation_id.trim()?r.conversation_id:null;
  if(!id){console.error(`record #${i+1}: no conversation_id; it cannot be reported.`);continue;}
  if(seen.has(id.trim())){errors.push({conversationId:id,error:'Duplicate conversation_id in the dataset.'});console.error(`${id}: duplicate conversation_id`);continue;}
  seen.add(id.trim());
  try{records.push(normalizeRecords([r])[0]);}catch(error){errors.push({conversationId:id,error:`Invalid record: ${error.message}`});console.error(`${id}: invalid record: ${error.message}`);}
}
const checklist=validateChecklist(await json(values.checklist));
if(!['model','rules'].includes(values.engine))throw new Error('engine must be model or rules');
if(values.engine==='model'&&(!process.env.MODEL_API_KEY||!process.env.MODEL_NAME))throw new Error('Set MODEL_API_KEY and MODEL_NAME. There is no silent fallback.');
const extraBody=parseExtraBody(process.env.MODEL_EXTRA_BODY);
const timeoutMs=Math.max(30,Math.min(300,Number(process.env.MODEL_TIMEOUT_SECONDS)||180))*1000;
const retries=values.engine==='model'?Math.max(0,Math.min(5,Number(values.retries)||0)):0;
// Configuration problems are not fixed by retrying the same request.
const FATAL=new Set(['MODEL_AUTH','MODEL_NOT_FOUND','MODEL_REDIRECT','MODEL_CONNECTION']);
const state={schemaVersion:1,records,checklists:[checklist],evaluations:[],appeals:[],createdAt:new Date().toISOString(),errors};
const out=resolve(values.out);await mkdir(dirname(out),{recursive:true});
await writeFile(out,JSON.stringify(state,null,2)+'\n');
for(const record of records){
  const started=Date.now();let lastError=null,done=false;
  for(let attempt=0;attempt<=retries&&!done;attempt++){
    try{
      const result=values.engine==='rules'?{results:evaluateRules(record,checklist),usage:null,verification:null}:await evaluateWithModel(record,checklist,{key:process.env.MODEL_API_KEY,model:process.env.MODEL_NAME,baseURL:process.env.MODEL_BASE_URL||'https://api.openai.com/v1',allowLocal:process.env.ALLOW_LOCAL_MODEL==='1',verificationMode:values.verification,timeoutMs,extraBody,inputPrice:Number(process.env.MODEL_INPUT_PRICE)||0,outputPrice:Number(process.env.MODEL_OUTPUT_PRICE)||0});
      const pending=result.results.filter(r=>r.answer==='pending').map(r=>r.criterionId);
      if(pending.length&&attempt<retries){lastError=new Error(`pending: ${pending.join(', ')}`);console.warn(`${record.conversation_id}: pending (${pending.join(', ')}), retrying`);continue;}
      const e={id:crypto.randomUUID(),conversationId:record.conversation_id,engine:values.engine,record,checklist,createdAt:new Date().toISOString(),durationMs:Date.now()-started,attempts:attempt+1,...result};
      e.originalScore=scoreResults(checklist,e.results);e.finalScore=e.originalScore;
      state.evaluations.push(e);done=true;
      console.log(`${record.conversation_id}: ${e.originalScore.total??'pending'}/100; calls=${e.usage?.calls||0}; ${((Date.now()-started)/1000).toFixed(1)}s`);
    }catch(error){
      lastError=error;
      if(FATAL.has(error.code)||attempt>=retries)break;
      console.warn(`${record.conversation_id}: ${error.message}; retrying`);
    }
  }
  if(!done){state.errors.push({conversationId:record.conversation_id,error:lastError?.message||'Unknown error.'});console.error(`${record.conversation_id}: ${lastError?.message}`);}
  await writeFile(out,JSON.stringify(state,null,2)+'\n');
  if(!done&&FATAL.has(lastError?.code)){console.error('Stopping: model configuration error affects every conversation. Fix .env and re-run.');for(const r of records.slice(records.indexOf(record)+1))state.errors.push({conversationId:r.conversation_id,error:`Not evaluated: ${lastError.message}`});await writeFile(out,JSON.stringify(state,null,2)+'\n');break;}
}
console.log(`Saved ${state.evaluations.length}/${raw.length} evaluations to ${out}; ${state.errors.length} errors`);
if(state.errors.length)process.exitCode=1;
