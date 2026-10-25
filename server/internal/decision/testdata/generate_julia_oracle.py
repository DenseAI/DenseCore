#!/usr/bin/env python3
"""Regenerate the Julia Go oracle in a restricted offline container.

Required read-only mounts: pinned source at /reference, installed Rust tokenizers
at /deps, owned repository scripts at /owned, julia-tokenizer-corpus.json at
/corpus.json, and this script at /oracle.py. Set PYTHONPATH=/deps:/owned.
No model modules are imported; only pinned pure validation/typed functions run.
Write stdout to an external artifact, then split cases/sequence_cases/named_cases
into the two JSON fixtures. The source revision and tokenizer hash are pinned.
"""
import ast,hashlib,json,math
from pathlib import Path
from tokenizers import Tokenizer
from julia_inputs import encode_request
for name,expected in {
 '/reference/julia_typed.py':'ed89e66a70fcedac1339347bd8fcca69fd2cd1a31535ad0148dae42c610b544d',
 '/reference/julia/data.py':'e3510fa4152ec11fa193046715991f44d7c2f85fd2488a98ef11c9d3db23da4e',
}.items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected
raw=Path('/reference/tokenizer/tokenizer.json').read_bytes()
sha=hashlib.sha256(raw).hexdigest()
assert sha=='609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f'
tok=Tokenizer.from_str(raw.decode())
texts=json.loads(Path('/corpus.json').read_text())['texts']
cases=[{'text':t,'ids':tok.encode(t,add_special_tokens=False).ids} for t in texts]
rows=[
 {'state':'A customer asks for a refund.','question':'Where should the request go?','options':['Billing','Technical support','Sales','Other']},
 {'state':{'日本語':'返品したい','n':3},'question':'対応する部署は？','options':['請求','技術','営業','その他'],'type':'choice'},
 {'state':['red','green'],'question':'The light is green.','options':['False','True'],'type':'noul'},
 {'state':'a'*200,'question':'Urgency?','options':['Low','Medium','High'],'type':'score'},
 {'state':'e\u0301 é Ａ ① 👩\u200d💻','question':'Choose a script','options':['한국어','العربية','English']},
 {'state':'\tfoo\r\nbar\u00a0x\u200by','question':'Whitespace?','options':['Present','Absent'],'type':'noul'},
 {'state':'reserved <mask>','question':'Marker?','options':['Yes','No']},
 {'state':'x','question':'Choose','options':['word '*70,'short']},
 {'state':'x','question':'word '*700,'options':['one','two']},
 {'state':'word '*9000,'question':'Context?','options':['one','two']},
 {'state':'<eos><bos><pad> ▁ literal','question':'Token?','options':['Reserved','Other']},
 {'state':'A customer asks for a refund.','question':'Where should the request go?','options':['Other','Sales','Technical support','Billing']},
]
sequences=[]
def append(row,limit,head,policy):
 item={'row':row,'max_length':limit,'head_length':head,'policy':policy}
 try:item['expected']=encode_request(tok,row,limit,head)
 except ValueError as e:item['error']=str(e)
 sequences.append(item)
for row in rows:append(row,8192,512,'checkpoint inference-policy.json')
for row in rows[:4]:
 for limit in (128,256,512):append(row,limit,64,'explicit supplementary short-context budget')
# Execute only the pinned pure typed/validation functions in the restricted
# oracle container; no module imports or model code from the reference run.
def function(path,name,namespace):
 source=ast.parse(Path(path).read_text())
 node=next(n for n in source.body if isinstance(n,ast.FunctionDef) and n.name==name)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
 return namespace[name]
validate=function('/reference/julia/data.py','validate_row',{'QTYPES':{'choice':0,'score':1,'noul':2}})
namespace={'math':math,'validate_row':validate}
typed=function('/reference/julia_typed.py','predict_typed',namespace)
class Capture:
 def logits(self,rows):
  self.rows=rows
  return [[0.0]*len(row['options']) for row in rows]
def request(question,state='hello'):
 return {'state':state,'questions':{'q':question}}
named=[
 request({'type':'choice','instructions':'Route?','criteria':{'billing_id':'Billing','tech_id':'Technical support'}}),
 request({'type':'score','instructions':'Urgency?','criteria':['Low','Medium','High']}),
 request({'type':'noul','instructions':'Safe?'}),
 request({'type':'noul','instructions':'Safe?','criteria':{'false':'Unsafe','true':'Safe'}}),
 request({'type':'choice','instructions':'部署は？','criteria':{'not_encoded_請求':'請求','not_encoded_技術':'技術'}},{'日本語':'返品したい','n':3}),
 request({'type':'score','instructions':'정도?','criteria':['낮음','높음']},['red','green']),
 request({'type':'noul','instructions':'Valid?','criteria':{'true':'Valid','false':'Invalid'}}),
 {'state':'shared state','questions':{'choice':{'type':'choice','instructions':'Route?','criteria':{'id_a':'Billing','id_b':'Tech'}},'score':{'type':'score','instructions':'Urgency?','criteria':['Low','High']},'noul':{'type':'noul','instructions':'Safe?'}}},
 request({'type':'choice','instructions':'Route?','criteria':{'id_a':'','id_b':'Tech'}}),
 request({'type':'score','instructions':'Urgency?','criteria':[1,2]}),
 request({'type':'noul','instructions':'Safe?','criteria':{'false':'Unsafe'}}),
 request({'type':'choice','instructions':'Route?','criteria':['Billing','Tech']}),
]
named_cases=[]
for req in named:
 item={'request':req,'max_length':8192,'head_length':512}
 engine=Capture()
 try:
  typed(engine,req['state'],req['questions'])
  item['rows']=engine.rows
  item['expected']=[encode_request(tok,row,8192,512) for row in engine.rows]
 except ValueError as e:item['error']=str(e)
 named_cases.append(item)
print(json.dumps({'tokenizer_sha256':sha,'revision':'a85b127321d580d65176c89ced8273f305745d85','reference':'Rust Hugging Face tokenizers; owned strict sequence helper checked against pinned upstream sequence; named rows captured from pinned pure typed.py function','typed_source_sha256':hashlib.sha256(Path('/reference/julia_typed.py').read_bytes()).hexdigest(),'cases':cases,'sequence_cases':sequences,'named_cases':named_cases},ensure_ascii=False))
