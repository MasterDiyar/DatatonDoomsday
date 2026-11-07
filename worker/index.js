import { handleAPI, StateRepository, d1Adapter } from '../src/core.mjs';
export default {
  async fetch(request,env){
    const url=new URL(request.url);
    let response;
    if(url.pathname.startsWith('/api/')){
      if(!env.DB)return new Response(JSON.stringify({error:'Хранилище не подключено.'}),{status:503,headers:{'Content-Type':'application/json'}});
      response=await handleAPI(request,new StateRepository(d1Adapter(env.DB)),env);
    }else{
      const asset=ASSETS[url.pathname];
      response=asset?new Response(asset.body,{headers:{'Content-Type':asset.type,'Cache-Control':'no-cache'}}):new Response('Not found',{status:404});
    }
    response.headers.set('X-Content-Type-Options','nosniff');
    response.headers.set('Referrer-Policy','no-referrer');
    response.headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src https: blob:; connect-src 'self'; frame-ancestors 'self' https://*.dynamics.com https://*.chatgpt.com https://chatgpt.com;");
    return response;
  }
};
