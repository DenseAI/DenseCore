#!/usr/bin/env python3
"""Independent pinned Julia PyTorch/native parity. Run in restricted container.

--prepare-only constructs the upstream model and exports exact nonpersistent
RoPE buffers, without any forward pass. Full verification requires the converted
GGUF and native library, and writes separately reusable numerical fixtures.
"""
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import struct

from download_julia_model import FILES, REVISION, verify


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gguf_tensors(path):
    """Independent reader for this converter's F32 GGUF profile."""
    with path.open('rb') as stream:
        def take(n):
            data=stream.read(n)
            if len(data)!=n: raise ValueError('truncated GGUF')
            return data
        def number(fmt): return struct.unpack(fmt,take(struct.calcsize(fmt)))[0]
        def string(): return take(number('<Q')).decode()
        magic,version,count,fields=struct.unpack('<4sIQQ',take(24))
        if magic!=b'GGUF' or version!=3: raise ValueError('invalid GGUF')
        metadata={}
        for _ in range(fields):
            key=string();kind=number('<I')
            if kind==8: value=string()
            elif kind==4: value=number('<I')
            elif kind==6: value=number('<f')
            else: raise ValueError('unsupported metadata type')
            metadata[key]=value
        tensors={}
        for _ in range(count):
            name=string();ndim=number('<I');shape=[number('<Q') for _ in range(ndim)]
            kind=number('<I');offset=number('<Q')
            if kind!=0 or name in tensors: raise ValueError('invalid F32 tensor descriptor')
            tensors[name]=(list(reversed(shape)),offset)
        data_offset=(stream.tell()+31)//32*32
    return metadata,tensors,data_offset


def compare_tensors(gguf,checkpoint,rope):
    import numpy as np
    from safetensors import safe_open
    metadata,tensors,offset=gguf_tensors(gguf)
    expected=set()
    with safe_open(str(checkpoint),framework='numpy') as state,gguf.open('rb') as stream:
        for name in state.keys():
            if name=='temperature':continue
            expected.add(name)
            source=state.get_tensor(name)
            shape,start=tensors[name]
            if shape!=list(source.shape):raise ValueError('GGUF tensor shape differs: '+name)
            stream.seek(offset+start)
            actual=np.frombuffer(stream.read(source.nbytes),dtype='<f4').reshape(shape)
            np.testing.assert_array_equal(actual,source,err_msg=name)
        for name,values in rope['frequencies'].items():
            expected.add(name);shape,start=tensors[name];stream.seek(offset+start)
            np.testing.assert_array_equal(np.frombuffer(stream.read(128),dtype='<f4'),np.array(values,dtype=np.float32))
        if expected!=set(tensors):raise ValueError('GGUF tensor set differs from checkpoint')
    if metadata['general.architecture']!='julia':raise ValueError('wrong GGUF architecture')
    return len(expected)-2


def fixtures(tokenizer,sequence):
    rows=[('choice',{'state':'A customer was charged twice for the same subscription.','question':'Which department should handle this request?','options':['Billing','Technical support','Sales','Other'],'type':'choice'}),
          ('score',{'state':'The answer explains the steps and cites supporting evidence.','question':'Rate answer quality.','options':['Poor','Adequate','Excellent'],'type':'score'}),
          ('noul_unicode',{'state':'서울에서 결제한 금액을 환불해 주세요. café 😊 日本語','question':'The customer requests a refund.','options':['false','true'],'type':'noul'}),
          ('local_attention',{'state':'An invoice was received and payment is due. '*35,'question':'Which department should handle this request?','options':['Billing','Technical support','Sales','Other'],'type':'choice'}),
          ('choice_reordered',{'state':'A customer was charged twice for the same subscription.','question':'Which department should handle this request?','options':['Other','Sales','Technical support','Billing'],'type':'choice'}),
          ('choice_after_long',{'state':'A customer was charged twice for the same subscription.','question':'Which department should handle this request?','options':['Billing','Technical support','Sales','Other'],'type':'choice'})]
    result=[]
    for name,row in rows:
        encoded=sequence(tokenizer,row,max_length=8192,head_length=512,strict=True)
        if name=='local_attention' and len(encoded['ids'])<=128:raise ValueError('long fixture is too short')
        result.append((name,row,encoded))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only',action='store_true')
    mode.add_argument('--reference-only',action='store_true',help='generate independent oracle fixtures without loading a native library')
    parser.add_argument('--gguf',type=Path)
    parser.add_argument('--library',type=Path)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--fixtures-output',type=Path)
    args=parser.parse_args()
    for name in ['model.safetensors','encoder/config.json','julia_config.json','julia/model.py','julia/data.py','julia/typed.py','inference-policy.json','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json']:
        if not verify(args.model/name,*FILES[name]):raise ValueError('unrecognized pinned input: '+name)
    import numpy as np
    import torch
    import transformers
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads);torch.set_grad_enabled(False)
    spec=importlib.util.spec_from_file_location('pinned_julia_model',args.model/'julia/model.py')
    upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)
    reference=upstream.JuliaDecisionModel.from_pretrained(args.model,memory_map=True)
    reference.eval();reference.encoder.config._attn_implementation='eager'
    rotary=reference.encoder.rotary_emb
    frequencies={'rope.freq_full':rotary.full_attention_inv_freq.float().tolist(),
                 'rope.freq_sliding':rotary.sliding_attention_inv_freq.float().tolist()}
    report={'source_revision':REVISION,'encoder_config_sha256':FILES['encoder/config.json'][1],
            'weights_sha256':FILES['model.safetensors'][1],'frequencies':frequencies,
            'versions':{'torch':torch.__version__,'transformers':transformers.__version__,'numpy':np.__version__},
            'precision':'FP32','threads':args.threads,'upstream_quality_gate':False,
            'inputs_sha256':{name:spec[1] for name,spec in FILES.items() if (args.model/name).is_file()}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.prepare_only:
        report['forward_executed']=False
        args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    if not args.gguf or (not args.reference_only and not args.library):
        parser.error('--gguf required; --library also required unless --reference-only')
    report['learned_tensors_exact']=compare_tensors(args.gguf,args.model/'model.safetensors',report)
    report['gguf_sha256']=sha(args.gguf)
    if not args.reference_only:report['library_sha256']=sha(args.library)
    report['checker_sha256']=sha(Path(__file__))
    report['platform']=platform.platform()
    report['limits']={'max_abs_logit':1e-3,'max_abs_probability':1e-4,'act_logit_atol':1e-3,'act_logit_rtol':1e-6,'max_abs_act_probability':1e-4}
    # Import only the pinned data module. This entire checker must execute in a
    # disposable restricted container because it executes external reference code.
    spec=importlib.util.spec_from_file_location('pinned_julia_data',args.model/'julia/data.py')
    data=importlib.util.module_from_spec(spec);spec.loader.exec_module(data)
    tokenizer=AutoTokenizer.from_pretrained(args.model/'tokenizer',trust_remote_code=False,local_files_only=True)
    lib=None;handle=None
    if not args.reference_only:
        lib=C.CDLL(str(args.library.resolve()))
        lib.DenseCoreDecisionLoad.argtypes=[C.c_char_p,C.c_int,C.c_char_p,C.c_size_t];lib.DenseCoreDecisionLoad.restype=C.c_void_p
        lib.DenseCoreDecisionFree.argtypes=[C.c_void_p]
        lib.DenseCoreDecisionPredict.argtypes=[C.c_void_p,C.POINTER(C.c_int32),C.c_size_t,C.c_int,C.POINTER(C.c_int32),C.c_size_t,C.POINTER(C.c_float),C.POINTER(C.c_float),C.c_char_p,C.c_size_t]
        lib.DenseCoreDecisionPredict.restype=C.c_int
        error=C.create_string_buffer(2048)
        handle=lib.DenseCoreDecisionLoad(str(args.gguf).encode(),args.threads,error,len(error))
        if not handle:raise RuntimeError(error.value.decode())
    report['cases']=[];records=[];passed=True
    try:
        for name,row,encoded in fixtures(tokenizer,data.sequence):
            tokens,markers,qtype=encoded['ids'],encoded['markers'],encoded['qtype']
            ids=torch.tensor([tokens]);positions=torch.tensor([markers])
            expected,acts=reference(ids,torch.ones_like(ids),positions,torch.ones_like(positions,dtype=torch.bool),torch.tensor([qtype]),return_actions=True)
            expected=expected[0].numpy();acts=acts[0].numpy()
            if args.reference_only:
                probability=torch.softmax(torch.from_numpy(expected),dim=0).numpy()
                act_probability=torch.softmax(torch.from_numpy(acts),dim=0).numpy()
                case={'name':name,'tokens':len(tokens),'markers':markers,
                      'reference_logits':expected.tolist(),'reference_act_logits':acts.tolist(),
                      'reference_probabilities':probability.tolist(),
                      'reference_act_probabilities':act_probability.tolist(),
                      'status':'reference_generated_native_not_executed'}
                report['cases'].append(case)
                records.append({'name':name,'row':row,'token_ids':tokens,'markers':markers,
                                'question_type':qtype,'reference_logits':expected.tolist(),
                                'reference_act_logits':acts.tolist(),
                                'reference_probabilities':probability.tolist(),
                                'reference_act_probabilities':act_probability.tolist(),'temperature':1.0})
                print(json.dumps(case),flush=True)
                continue
            observed=(C.c_float*len(markers))();observed_act=(C.c_float*2)()
            if not lib.DenseCoreDecisionPredict(handle,(C.c_int32*len(tokens))(*tokens),len(tokens),qtype,(C.c_int32*len(markers))(*markers),len(markers),observed,observed_act,error,len(error)):
                raise RuntimeError(error.value.decode())
            observed=np.array(observed,dtype=np.float32);observed_act=np.array(observed_act,dtype=np.float32)
            softmax=lambda z:torch.softmax(torch.from_numpy(z),dim=0).numpy()
            logit_error=float(np.max(np.abs(observed-expected)));prob_error=float(np.max(np.abs(softmax(observed)-softmax(expected))))
            act_error=float(np.max(np.abs(observed_act-acts)));act_prob_error=float(np.max(np.abs(softmax(observed_act)-softmax(acts))))
            success=logit_error<=1e-3 and prob_error<=1e-4 and bool(np.all(np.abs(observed_act-acts)<=1e-3+1e-6*np.abs(acts))) and act_prob_error<=1e-4
            passed &= success
            case={'name':name,'tokens':len(tokens),'markers':markers,'max_abs_logit':logit_error,'max_abs_probability':prob_error,'max_abs_act_logit':act_error,'max_abs_act_probability':act_prob_error,'reference_logits':expected.tolist(),'native_logits':observed.tolist(),'reference_act_logits':acts.tolist(),'native_act_logits':observed_act.tolist(),'passed':success}
            report['cases'].append(case)
            records.append({'name':name,'row':row,'token_ids':tokens,'markers':markers,'question_type':qtype,'reference_logits':expected.tolist(),'reference_act_logits':acts.tolist(),'reference_probabilities':softmax(expected).tolist(),'temperature':1.0})
            print(json.dumps(case),flush=True)
    finally:
        if lib is not None:lib.DenseCoreDecisionFree(handle)
    if args.reference_only:
        report['status']='reference_generated_native_not_executed'
        report['native_parity']='not_executed'
    else:
        report['passed']=passed
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    if args.fixtures_output:
        args.fixtures_output.parent.mkdir(parents=True,exist_ok=True)
        args.fixtures_output.write_text(json.dumps({'repository':'SupersonicLabs/Julia-1','revision':REVISION,'weights_sha256':FILES['model.safetensors'][1],'tokenizer_sha256':FILES['tokenizer/tokenizer.json'][1],'cases':records},ensure_ascii=False,indent=2)+'\n')
    if not args.reference_only and not passed:raise SystemExit('Julia parity FAILED; see report')


if __name__=='__main__':main()
