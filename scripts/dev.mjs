import { createServer } from 'node:http';
import { DatabaseSync } from 'node:sqlite';
import { readFile, mkdir } from 'node:fs/promises';
import { resolve, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { handleAPI, StateRepository } from '../src/core.mjs';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
await mkdir(resolve(root,'.data'),{recursive:true});
const db=new DatabaseSync(resolve(root,'.data/cqa.sqlite'));
db.exec('PRAGMA journal_mode=WAL;');
db.exec(await readFile(resolve(root,'drizzle/0000_cqa_state.sql'),'utf8'));
const repository=new StateRepository({
  get:async()=>db.prepare('SELECT state_json,revision FROM cqa_state WHERE id=?').get('main'),
  insert:async(json,date)=>db.prepare('INSERT OR IGNORE INTO cqa_state (id,state_json,revision,updated_at) VALUES (?,?,0,?)').run('main',json,date),
  update:async(json,rev,date)=>db.prepare('UPDATE cqa_state SET state_json=?,revision=revision+1,updated_at=? WHERE id=? AND revision=?').run(json,date,'main',rev).changes>0
});
const types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml'};
const server=createServer(async(req,res)=>{
  try {
    let length=0;const chunks=[];
    for await(const chunk of req){length+=chunk.length;if(length>6*1024*1024){res.writeHead(413,{'Content-Type':'application/json'});res.end(JSON.stringify({error:'Запрос больше 6 МБ.'}));return;}chunks.push(chunk);}
    const url=new URL(req.url,`http://${req.headers.host}`);
    let response;
    if(url.pathname.startsWith('/api/')){
      response=await handleAPI(new Request(url,{method:req.method,headers:req.headers,body:chunks.length?Buffer.concat(chunks):undefined}),repository,{...process.env,STORAGE_KIND:'SQLite'});
    }else{
      const allowed={'/':'index.html','/index.html':'index.html','/app.js':'app.js','/style.css':'style.css','/favicon.svg':'favicon.svg'};
      const filename=allowed[url.pathname];
      response=filename?new Response(await readFile(resolve(root,'public',filename)),{headers:{'Content-Type':types[extname(filename)],'Cache-Control':'no-cache'}}):new Response('Not found',{status:404});
    }
    response.headers.set('X-Content-Type-Options','nosniff');response.headers.set('Referrer-Policy','no-referrer');
    res.writeHead(response.status,Object.fromEntries(response.headers));res.end(Buffer.from(await response.arrayBuffer()));
  }catch(e){res.writeHead(500,{'Content-Type':'application/json'});res.end(JSON.stringify({error:'Ошибка локального сервера.'}));console.error(e.message);}
});
const port=Number(process.env.PORT||4173);
server.listen(port,'127.0.0.1',()=>console.log(`CQA Studio: http://127.0.0.1:${port}`));
for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>server.close(()=>{db.close();process.exit(0);}));
