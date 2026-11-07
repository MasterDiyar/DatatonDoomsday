import test from 'node:test';
import assert from 'node:assert/strict';
import {defaultChecklist,buildModelMessages,evaluateWithModel,parseExtraBody,validateModelResults} from '../src/core.mjs';
import {buildSubmission,validateSubmission,toSubmissionEntry,datasetIds,CRITERION_MAP,SUBMISSION_ORDER} from '../src/hackathon-export.mjs';

const record={conversation_id:'demo-001',channel:'agent chat',language:'EN',audio_url:null,transcript:[
  {speaker:'client',timestamp:'00:00',text:'What is the commission for US stocks?'},
  {speaker:'bot',timestamp:'00:01',text:'Connecting you to an operator.'},
  {speaker:'operator',timestamp:'00:20',text:'Hello! My name is Anna.'},
  {speaker:'operator',timestamp:'00:40',text:'The commission is listed on the tariffs page.'},
  {speaker:'client',timestamp:'01:00',text:'Thanks.'},
  {speaker:'operator',timestamp:'01:10',text:'Anything else? Goodbye!'}
]};
// Answers of the fictional example on page 2 of the results appendix (final_score 85.38).
const guideAnswers={contact:true,needs:null,identification:null,consultation:15,empathy:true,management:7,closing:true,gross_errors:false};
const results=answers=>Object.entries(answers).map(([criterionId,answer])=>({criterionId,answer,explanation:`Test rationale ${criterionId}.`,evidence:answer===null?[]:[2],contextEvidence:answer===null?[0]:[],checkedRange:{start:0,end:5},confidence:0.9}));
const evaluation=(answers=guideAnswers,id='demo-001')=>({id:`e-${id}`,conversationId:id,engine:'model',record:{...record,conversation_id:id},checklist:defaultChecklist(),results:results(answers)});
const build=(state,extra={})=>buildSubmission({state,teamName:'Doomsday',modelName:'Qwen2.5-7B-Instruct + CQA LoRA',mode:'local',...extra});

test('appendix example: ID mapping, N/A redistribution and 85.38 total',()=>{
  const entry=toSubmissionEntry(evaluation());
  assert.equal(entry.status,'ok');
  assert.deepEqual(entry.criteria.map(c=>c.criterion_id),SUBMISSION_ORDER);
  const v=Object.fromEntries(entry.criteria.map(c=>[c.criterion_id,c.value]));
  assert.deepEqual(v,{contact:true,discovery:null,identification:null,consultation:15,empathy:true,dialogue:7,closure:true,critical:false});
  assert.deepEqual(entry.score,{final_score:85.38,critical_flag:false});
});
test('evidence_turns are 1-based over the original transcript, context included',()=>{
  const entry=toSubmissionEntry(evaluation());
  assert.deepEqual(entry.criteria.find(c=>c.criterion_id==='contact').evidence_turns,[3]);
  assert.deepEqual(entry.criteria.find(c=>c.criterion_id==='discovery').evidence_turns,[1]);
});
test('missing action without evidence points at inspected operator turns',()=>{
  const e=evaluation();const r=e.results.find(r=>r.criterionId==='closing');r.answer=false;r.evidence=[];r.checkedRange={start:4,end:5};
  assert.deepEqual(toSubmissionEntry(e).criteria.find(c=>c.criterion_id==='closure').evidence_turns,[6]);
});
test('critical flag mirrors the critical criterion and does not zero the score',()=>{
  const entry=toSubmissionEntry(evaluation({...guideAnswers,gross_errors:true}));
  assert.equal(entry.score.critical_flag,true);
  assert.equal(entry.criteria.find(c=>c.criterion_id==='critical').value,true);
  assert.equal(entry.score.final_score,85.38);
});
test('all applicable full marks give exactly 100',()=>{
  const entry=toSubmissionEntry(evaluation({contact:true,needs:true,identification:true,consultation:20,empathy:true,management:10,closing:true,gross_errors:false}));
  assert.equal(entry.score.final_score,100);
});
test('failed, pending and unevaluated conversations become error entries, never zeros',()=>{
  const state={records:[],evaluations:[evaluation(),evaluation({...guideAnswers,consultation:'pending'},'p-1')],errors:[{conversationId:'x-1',error:'Model timeout.'}]};
  const file=build(state,{ids:['demo-001','p-1','x-1','never']});
  assert.deepEqual(file.results.map(r=>r.status),['ok','error','error','error']);
  for(const r of file.results.slice(1)){assert.deepEqual(r.criteria,[]);assert.equal(r.score,null);assert.ok(r.error);}
  assert.match(file.results[1].error,/consultation/);
  assert.equal(file.results[2].error,'Model timeout.');
  assert.deepEqual(validateSubmission(file,{expectIds:['demo-001','p-1','x-1','never'],expectCount:4}),[]);
});
test('latest evaluation of the current transcript wins; rules-engine results are not exported as model',()=>{
  const old=evaluation({...guideAnswers,consultation:5});const rules={...evaluation(),engine:'rules'};
  const file=build({records:[record],evaluations:[old,evaluation(),rules],errors:[]});
  assert.equal(file.results.length,1);
  assert.equal(file.results[0].criteria.find(c=>c.criterion_id==='consultation').value,15);
});
test('validator rejects duplicates, missing IDs, wrong counts, bad values and 0-based turns',()=>{
  const file=build({records:[],evaluations:[evaluation()],errors:[]},{ids:['demo-001']});
  assert.deepEqual(validateSubmission(file,{expectIds:['demo-001']}),[]);
  const dup={...file,results:[file.results[0],file.results[0]]};
  assert.ok(validateSubmission(dup).some(p=>/duplicate/.test(p)));
  assert.ok(validateSubmission(file,{expectIds:['demo-001','demo-002']}).some(p=>/Missing conversation_id demo-002/.test(p)));
  assert.ok(validateSubmission(file,{expectCount:10}).some(p=>/exactly 10/.test(p)));
  const bad=structuredClone(file);bad.results[0].criteria[3].value=12;bad.results[0].criteria[0].evidence_turns=[0];
  const problems=validateSubmission(bad);
  assert.ok(problems.some(p=>/invalid value for consultation/.test(p)));
  assert.ok(problems.some(p=>/1-based/.test(p)));
  const flag=structuredClone(file);flag.results[0].score.critical_flag=true;
  assert.ok(validateSubmission(flag).some(p=>/critical_flag/.test(p)));
});
test('dataset IDs are read from organiser payload shapes without validation',()=>{
  assert.deepEqual(datasetIds({data:{records:[{conversation_id:'a'},{conversation_id:'b',transcript:'broken'}]}}),['a','b']);
  assert.deepEqual(datasetIds([{conversation_id:' c '}]),['c']);
});
test('criterion map covers every official checklist criterion',()=>{
  assert.deepEqual(Object.keys(CRITERION_MAP).sort(),defaultChecklist().criteria.map(c=>c.id).sort());
});
test('training prompts are identical to the inference request and targets pass validation',async()=>{
  const checklist=defaultChecklist();let sent;
  const target={results:results(guideAnswers).map(({confidence,...r})=>r)};
  const fetcher=async(url,init)=>{sent=JSON.parse(init.body);return new Response(JSON.stringify({choices:[{message:{content:JSON.stringify(target)}}],usage:{}}));};
  await evaluateWithModel(record,checklist,{baseURL:'https://model.example.org/v1',key:'k',model:'m',verificationMode:'single',extraBody:{temperature:0,model:'ignored'}},fetcher);
  assert.deepEqual(sent.messages,buildModelMessages(record,checklist));
  assert.equal(sent.temperature,0);assert.equal(sent.model,'m');
  assert.equal(validateModelResults(target,record,checklist).length,8);
});
test('MODEL_EXTRA_BODY must be a JSON object and cannot override protocol fields',()=>{
  assert.deepEqual(parseExtraBody(''),{});
  assert.deepEqual(parseExtraBody('{"temperature":0.3,"messages":[],"response_format":null}'),{temperature:0.3});
  assert.throws(()=>parseExtraBody('[1]'));assert.throws(()=>parseExtraBody('nope'));
});
