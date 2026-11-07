// Generate synthetic RU/KZ/EN broker support chats with controlled operator behaviour,
// to augment the small DEV set before fine-tuning. Each dialogue is created from a scenario
// (language, topic, behaviours); behaviours that unambiguously imply a checklist answer are
// written to a sidecar file and later used by build-sft-dataset.mjs to drop teacher labels
// that contradict the scenario.
//
// The generator is any OpenAI-compatible API: TEACHER_* variables, falling back to MODEL_*.
// Output dialogues are fictional and must still be labelled (scripts/evaluate-file.mjs).
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {parseArgs} from 'node:util';
import {normalizeRecords,safeModelURL} from '../src/core.mjs';

const {values}=parseArgs({options:{
  count:{type:'string',default:'300'},out:{type:'string',default:'datasets/synth-dialogues.json'},
  'scenarios-out':{type:'string',default:'datasets/synth-scenarios.json'},seed:{type:'string',default:'1'},
  concurrency:{type:'string',default:'3'},prefix:{type:'string',default:'synth-'},resume:{type:'boolean',default:true}
}});
const env=process.env;
const key=env.TEACHER_API_KEY||env.MODEL_API_KEY,model=env.TEACHER_MODEL_NAME||env.MODEL_NAME;
const baseURL=(env.TEACHER_BASE_URL||env.MODEL_BASE_URL||'https://api.openai.com/v1').replace(/\/$/,'');
if(!key||!model){console.error('Set TEACHER_API_KEY and TEACHER_MODEL_NAME (or MODEL_API_KEY / MODEL_NAME).');process.exit(1);}
const endpoint=safeModelURL(`${baseURL}/chat/completions`,env.ALLOW_LOCAL_MODEL==='1');

// Small deterministic PRNG so the scenario list is reproducible for a given seed.
let s=Number(values.seed)>>>0||1;const rand=()=>((s=Math.imul(s^(s>>>15),2246822507)^Math.imul(s^(s>>>13),3266489909),s^=s>>>16)>>>0)/4294967296;
const pick=a=>a[Math.floor(rand()*a.length)];

const TOPICS={
  general:['commission and tariff plans','how to open a brokerage account','trading hours of US and KASE exchanges','what documents are needed for account opening','how dividends are credited','difference between market and limit orders','how to install and log in to the mobile app'],
  secure:['client asks for their current portfolio balance','client wants to withdraw money to a new bank card','client asks why a specific order of theirs was not executed','client wants to change the phone number linked to the account','client asks for a statement of their transactions','client forgot password and wants a reset','client asks about a blocked account']
};
// Behaviour → instruction for the generator, and checklist answers it unambiguously implies
// (internal criterion IDs). Graded levels (consultation/management) are left to the labeller.
const B={
  good_open:{text:'In the FIRST operator message the operator greets the client in the client\'s language AND introduces themselves by name.',expect:{contact:true}},
  no_intro:{text:'In the first operator message the operator greets but does NOT introduce themselves (no name), and never introduces later.',expect:{contact:false}},
  late_intro:{text:'The first operator message jumps straight to business without greeting; the operator introduces themselves only in a later message.',expect:{contact:false}},
  full_info:{text:'The client states the question completely and clearly in the very first message; nothing needs clarification.',expect:{needs:null}},
  clarify_done:{text:'The client\'s first message is vague; the operator asks a targeted clarifying question before answering.',expect:{needs:true}},
  clarify_skipped:{text:'The client\'s first message is vague and ambiguous; the operator does NOT clarify and answers a guess.',expect:{needs:false}},
  id_not_needed:{text:'The request is a general question that requires no identity verification; no identification takes place.',expect:{identification:null}},
  id_done:{text:'Before disclosing account data or performing the action, the operator asks the client to verify identity; a SYSTEM event confirms successful identification; only then the operator proceeds.',expect:{identification:true}},
  id_skipped:{text:'The operator discloses account data / performs the secure action WITHOUT any identity verification.',expect:{identification:false}},
  answer_full:{text:'The operator gives a correct, complete answer or a correct next step.'},
  answer_partial:{text:'The client asks two important questions; the operator answers only one and leaves the other substantively open.'},
  answer_wrong:{text:'The operator\'s key recommendation is clearly wrong or meaningless.'},
  client_leaves:{text:'The client greets, then disconnects before ever formulating any request.',expect:{consultation:null}},
  complaint_handled:{text:'The client is frustrated about a specific inconvenience; the operator is polite and explicitly responds to that specific problem.',expect:{empathy:true}},
  complaint_ignored:{text:'The client is frustrated about a specific inconvenience; the operator ignores the frustration and answers dryly.',expect:{empathy:false}},
  clear_flow:{text:'Operator messages are clear and coherent; when the operator needs time to check something, they warn the client and the next message comes a few minutes later.'},
  silent_wait:{text:'At one point the operator goes silent for 8+ minutes (visible in timestamps) without warning; the client asks if anyone is there.'},
  confusing:{text:'Operator messages are contradictory and incoherent, so the client cannot understand what to do.'},
  good_close:{text:'At the end the operator summarises or offers further help AND says goodbye.',expect:{closing:true}},
  no_goodbye:{text:'After answering, the operator ends the chat without any goodbye; a SYSTEM event says the chat was closed by the operator.',expect:{closing:false}},
  disconnect:{text:'The chat is cut off abruptly by a SYSTEM disconnect event before it can be properly completed.',expect:{closing:null}},
  clean:{text:'The operator is never insulting, does not offer private paid services and does not disclose confidential information.',expect:{gross_errors:false}},
  rude:{text:'At one point the operator insults or belittles the client.',expect:{gross_errors:true}},
  private_offer:{text:'The operator suggests the client transfer money directly to the operator for investing, or asks for the client\'s personal phone number to give paid consultations privately.',expect:{gross_errors:true}},
  leak:{text:'The operator discloses confidential information (e.g. another client\'s data or internal confidential details).',expect:{gross_errors:true}}
};
const weighted=pairs=>{const total=pairs.reduce((a,[,w])=>a+w,0);let r=rand()*total;for(const [k,w] of pairs){if((r-=w)<0)return k;}return pairs.at(-1)[0];};
function scenario(i){
  const language=weighted([['RU',0.4],['KZ',0.3],['EN',0.3]]);
  const secure=rand()<0.45;
  const behaviours=[];
  const leaves=rand()<0.05;
  behaviours.push(weighted([['good_open',0.6],['no_intro',0.25],['late_intro',0.15]]));
  if(leaves){behaviours.push('client_leaves','clean');}
  else{
    behaviours.push(weighted([['full_info',0.4],['clarify_done',0.4],['clarify_skipped',0.2]]));
    behaviours.push(secure?weighted([['id_done',0.6],['id_skipped',0.4]]):'id_not_needed');
    behaviours.push(weighted([['answer_full',0.6],['answer_partial',0.25],['answer_wrong',0.15]]));
    if(rand()<0.45)behaviours.push(weighted([['complaint_handled',0.6],['complaint_ignored',0.4]]));
    behaviours.push(weighted([['clear_flow',0.65],['silent_wait',0.2],['confusing',0.15]]));
    behaviours.push(weighted([['good_close',0.6],['no_goodbye',0.25],['disconnect',0.15]]));
    behaviours.push(weighted([['clean',0.82],['rude',0.06],['private_offer',0.07],['leak',0.05]]));
  }
  const topic=pick(secure?TOPICS.secure:TOPICS.general);
  const withBot=rand()<0.5;
  const expected=Object.assign({},...behaviours.map(b=>B[b].expect||{}));
  return {id:`${values.prefix}${String(i+1).padStart(4,'0')}`,language,topic,secure,withBot,behaviours,expected};
}
const LANG={RU:'Russian',KZ:'Kazakh (Cyrillic script; natural code-switching with Russian is acceptable for terms)',EN:'English'};
async function generate(sc){
  const instruction=`Write a realistic, fictional live-chat transcript between a client and a support operator of an investment broker (fictional company "Freedom Broker"). No real personal data.
Language of client and operator: ${LANG[sc.language]}.
Topic: ${sc.topic}.
${sc.withBot?'The chat starts with an automated BOT greeting/routing message before the operator joins.':'There is no bot.'}
Operator behaviour (must all be clearly visible in the text):
${sc.behaviours.map(b=>`- ${B[b].text}`).join('\n')}
Rules: 6–20 turns. Speakers are exactly "client", "operator", "bot" or "system". SYSTEM turns are short platform events (e.g. identification confirmed, chat closed, client disconnected) and may be in Russian or English. Timestamps are ISO 8601 UTC, increasing, realistic (seconds to minutes apart). Do not label or mention the behaviours or any quality checklist inside the chat.
Return only JSON: {"transcript":[{"timestamp":"2025-03-04T10:00:00Z","speaker":"client","text":"..."}]}`;
  const response=await fetch(endpoint.href,{method:'POST',redirect:'manual',signal:AbortSignal.timeout(180000),headers:{'Content-Type':'application/json',Authorization:`Bearer ${key}`},
    body:JSON.stringify({model,temperature:0.9,response_format:{type:'json_object'},messages:[{role:'user',content:instruction}]})});
  if(!response.ok)throw new Error(`HTTP ${response.status}`);
  const completion=await response.json();
  const parsed=JSON.parse(completion.choices?.[0]?.message?.content||'{}');
  const [record]=normalizeRecords([{conversation_id:sc.id,language:sc.language,channel:'agent chat',transcript:parsed.transcript}]);
  if(!record.transcript.some(m=>m.speaker==='operator'))throw new Error('no operator turns');
  return record;
}

const count=Number(values.count),out=resolve(values.out),scOut=resolve(values['scenarios-out']);
await mkdir(dirname(out),{recursive:true});await mkdir(dirname(scOut),{recursive:true});
const load=async(path,fallback)=>{try{return JSON.parse(await readFile(path,'utf8'));}catch{return fallback;}};
const existing=values.resume?await load(out,{records:[]}):{records:[]};
const scenarios=values.resume?await load(scOut,{}):{};
const done=new Set(existing.records.map(r=>r.conversation_id));
const queue=Array.from({length:count},(_,i)=>scenario(i)).filter(sc=>!done.has(sc.id));
let failures=0,writing=Promise.resolve();
// Serialise file writes: several workers share the two output files.
const save=()=>writing=writing.then(async()=>{existing.records.sort((a,b)=>a.conversation_id.localeCompare(b.conversation_id));await writeFile(out,JSON.stringify(existing,null,2)+'\n');await writeFile(scOut,JSON.stringify(scenarios,null,2)+'\n');});
async function worker(){
  while(queue.length){
    const sc=queue.shift();
    for(let attempt=1;attempt<=2;attempt++){
      try{const record=await generate(sc);existing.records.push(record);scenarios[sc.id]=sc;console.log(`${sc.id} ${sc.language} ${sc.behaviours.join(',')}`);break;}
      catch(error){if(attempt===2){failures++;console.warn(`${sc.id}: ${error.message}`);}}
    }
    await save();
  }
}
await Promise.all(Array.from({length:Math.max(1,Number(values.concurrency))},worker));
console.log(`Saved ${existing.records.length} dialogues to ${out} (${failures} failed); scenarios in ${scOut}`);
