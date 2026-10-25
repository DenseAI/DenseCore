#!/usr/bin/env python3
"""Independent preprocessing check; execute only in a restricted container."""
import argparse,ast,hashlib,json
from pathlib import Path
from download_julia_model import FILES,verify
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,required=True)
args=parser.parse_args()
for name in ['tokenizer/tokenizer.json','julia/data.py']:
 if not verify(args.reference/name,*FILES[name]):
  raise SystemExit('Invalid pinned reference: '+name)
from tokenizers import Tokenizer
from julia_inputs import encode_request,probabilities
raw=(args.reference/'tokenizer/tokenizer.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='609d8f4c067cd3950f88594c5a802616cea245823836ef5848ee4fc40aab5b6f'
tokenizer=Tokenizer.from_str(raw.decode())
class Adapter:
 mask_token='<mask>';mask_token_id=4;cls_token_id=2;sep_token_id=1
 def __call__(self,text,add_special_tokens=False):return {'input_ids':tokenizer.encode(text,add_special_tokens=add_special_tokens).ids}
source=ast.parse((args.reference/'julia/data.py').read_text())
sequence=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='sequence')
namespace={'json':json,'QTYPES':{'choice':0,'score':1,'noul':2}}
exec(compile(ast.Module(body=[sequence],type_ignores=[]),'<pinned upstream sequence>','exec'),namespace)
rows=[{'state':'A customer asks for a refund.','question':'Where should the request go?','options':['Billing','Technical support','Sales','Other']},{'state':{'日本語':'返品したい','n':3},'question':'対応する部署は？','options':['請求','技術','営業','その他'],'type':'choice'},{'state':['red','green'],'question':'The light is green.','options':['False','True'],'type':'noul'},{'state':'a'*200,'question':'Urgency?','options':['Low','Medium','High'],'type':'score'}]
passed=0
for row in rows:
 for limit in [128,256,512]:
  head=min(64,limit-8)
  try: expected=namespace['sequence'](Adapter(),row,limit,head,strict=True)
  except ValueError:
   try:encode_request(tokenizer,row,limit,head)
   except ValueError:passed+=1;continue
   raise AssertionError('upstream rejection not preserved')
  actual=encode_request(tokenizer,row,limit,head)
  assert {key:actual[key] for key in expected}==expected,(expected,actual)
  assert actual['valid_input_tokens']==len(expected['ids'])
  passed+=1
print(json.dumps({'tokenizer_sha256':hashlib.sha256(raw).hexdigest(),'reference_revision':'a85b127321d580d65176c89ced8273f305745d85','sequence_cases_passed':passed,'native_model_parity':'unavailable'}))
