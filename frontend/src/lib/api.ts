import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { errorMessage } from './utils';
const tokenKey='spark-cluster.admin-token';
let prompt: ((message:string)=>Promise<string>) | null=null;
export function installAuthPrompt(handler:typeof prompt){prompt=handler;return ()=>{prompt=null;};}
export function storedToken(){try{return localStorage.getItem(tokenKey)||'';}catch{return '';}}
export function clearToken(){try{localStorage.removeItem(tokenKey);}catch{ /* storage unavailable */ }}
export function saveToken(token:string){try{localStorage.setItem(tokenKey,token);}catch{ /* session-only browser */ }}
export class ApiError extends Error{constructor(message:string,public status:number,public payload:unknown){super(message);}}
export type ApiOptions=Omit<RequestInit,'body'> & {body?:unknown;interactive?:boolean};
export async function api<T>(path:string,{body,interactive=false,...options}:ApiOptions={}):Promise<T>{
  let candidate=new Headers(options.headers).get('Authorization')?.replace(/^Bearer\s+/i,'').trim()||storedToken();
  for(;;){
    const response=await fetch(path,{...options,body:body===undefined?undefined:JSON.stringify(body),headers:{'Content-Type':'application/json',...options.headers,...(candidate?{Authorization:`Bearer ${candidate}`}:{})}});
    const data:unknown=await response.json().catch(()=>({}));
    if(response.status===401){if(candidate===storedToken())clearToken();if(interactive&&prompt){candidate=await prompt(candidate?'The saved token was rejected. Enter the current administration token.':'Enter the administration token to continue.');continue;}}
    if(!response.ok){const detail=(data as {detail?:unknown}).detail;const message=typeof detail==='string'?detail:Array.isArray(detail)?detail.map(d=>(d as {msg:string}).msg).join('; '):`Request failed (${response.status})`;throw new ApiError(message,response.status,data);}
    return data as T;
  }
}
export function useApi<T>(path:string|null,interval?:number){return useQuery<T>({queryKey:[path],queryFn:({signal})=>api<T>(path!,{signal}),enabled:!!path,refetchInterval:interval,refetchIntervalInBackground:false,retry:false,staleTime:1000});}
export function useAction(){const client=useQueryClient();const[busy,setBusy]=useState(false);const[error,setError]=useState('');const[message,setMessage]=useState('');async function run<T>(action:()=>Promise<T>,success?:string):Promise<T|undefined>{if(busy)return;setBusy(true);setError('');setMessage('');try{const result=await action();if(success)setMessage(success);await client.invalidateQueries();return result;}catch(e){setError(errorMessage(e));return undefined;}finally{setBusy(false);}}return{busy,error,message,run,setError,setMessage};}
export const post=<T,>(path:string,body?:unknown)=>api<T>(path,{method:'POST',body,interactive:true});
