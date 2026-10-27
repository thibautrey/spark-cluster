import {readdir, readFile, writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
const directory=resolve(import.meta.dirname,'../../src/spark_cluster/static/assets');
for(const name of await readdir(directory)){
  if(!/\.(?:js|css)$/.test(name))continue;
  const path=resolve(directory,name);
  const source=await readFile(path,'utf8');
  const clean=source.replace(/[ \t]+$/gm,'');
  if(clean!==source)await writeFile(path,clean);
}
