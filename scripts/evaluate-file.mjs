import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {normalizeRecords,validateChecklist,evaluateRules,evaluateWithModel,scoreResults} from '../src/core.mjs';
const {values}=parseArgs({options:{data:{type:'string'},checklist:{type:'string'},out:{type:'string',default:'results.json'},engine:{type:'string',default:'model'},verification:{type:'string',default:'verified'}}});
if(!values.data||!values.checklist){console.error('Usage: node --env-file-if-exists=.env scripts/evaluate-file.mjs --data control.json --checklist official.json --out results.json [--engine rules] [--verification single]');process.exit(1);}
const json=async path=>JSON.parse(await readFile(resolve(path),'utf8'));
const records=normalizeRecords(await json(values.data));
const checklist=validateChecklist(await json(values.checklist));
if(!['model','rules'].includes(values.engine))throw new Error('engine must be model or rules');
if(values.engine==='model'&&(!process.env.MODEL_API_KEY||!process.env.MODEL_NAME))throw new Error('Set MODEL_API_KEY and MODEL_NAME. There is no silent fallback.');
const state={schemaVersion:1,records,checklists:[checklist],evaluations:[],appeals:[],createdAt:new Date().toISOString(),errors:[]};
const out=resolve(values.out);await mkdir(dirname(out),{recursive:true});
for(const record of records){
  try{
    const result=values.engine==='rules'?{results:evaluateRules(record,checklist),usage:null,verification:null}:await evaluateWithModel(record,checklist,{key:process.env.MODEL_API_KEY,model:process.env.MODEL_NAME,baseURL:process.env.MODEL_BASE_URL||'https://api.openai.com/v1',allowLocal:process.env.ALLOW_LOCAL_MODEL==='1',verificationMode:values.verification,inputPrice:Number(process.env.MODEL_INPUT_PRICE)||0,outputPrice:Number(process.env.MODEL_OUTPUT_PRICE)||0});
    const e={id:crypto.randomUUID(),conversationId:record.conversation_id,engine:values.engine,record,checklist,createdAt:new Date().toISOString(),...result};
    e.originalScore=scoreResults(checklist,e.results);e.finalScore=e.originalScore;
    state.evaluations.push(e);console.log(`${record.conversation_id}: ${e.originalScore.total??'pending'}/100; calls=${e.usage?.calls||0}`);
  }catch(error){state.errors.push({conversationId:record.conversation_id,error:error.message});console.error(`${record.conversation_id}: ${error.message}`);}
  await writeFile(out,JSON.stringify(state,null,2)+'\n');
}
console.log(`Saved ${state.evaluations.length}/${records.length} evaluations to ${out}`);
if(state.errors.length)process.exitCode=1;
