import {readFile,writeFile} from 'node:fs/promises';
import {parseArgs} from 'node:util';
import {compareGroundTruth} from '../src/core.mjs';
const {values}=parseArgs({options:{results:{type:'string'},truth:{type:'string'},out:{type:'string',default:'accuracy-report.json'}}});
if(!values.results||!values.truth){console.error('Usage: node scripts/benchmark.mjs --results results.json --truth ground-truth.json --out accuracy-report.json');process.exit(1);}
const state=JSON.parse(await readFile(values.results,'utf8')),truth=JSON.parse(await readFile(values.truth,'utf8'));
const report=compareGroundTruth(state,truth);await writeFile(values.out,JSON.stringify(report,null,2)+'\n');
console.log(`Exact answers: ${(report.exactAccuracy*100).toFixed(2)}%; normalized MAE: ${(report.normalizedMAE*100).toFixed(2)}%; score MAE: ${report.totalMAE??'incomplete truth'}/100`);
