"""Create deterministic, labelled calibration/evaluation sets for CV and text."""
import argparse
import hashlib
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .common import ARTIFACTS, CACHE, DATA, ROOT, SEED, SEQ, read_json, sha256, tensorflow, write_json


def balanced_selection(records, labels, per_class, seed=SEED):
    rng=np.random.default_rng(seed)
    chosen=[]
    for label in sorted(set(labels)):
        candidates=[r for r,y in zip(records,labels) if y==label]
        if len(candidates)<per_class: raise ValueError('Too few examples for balanced selection')
        chosen.extend(candidates[i] for i in rng.choice(len(candidates),per_class,replace=False))
    return chosen


def normalize_text(text):
    return ' '.join(text.lower().split())


def cv_preprocess(path):
    with Image.open(path) as image:
        pixels=np.asarray(ImageOps.exif_transpose(image).convert('RGB').resize(
            (224,224),Image.Resampling.BILINEAR),dtype=np.float32)
    # Keras ResNet50 caffe preprocessing: BGR minus channel means.
    return np.ascontiguousarray(pixels[:,:,::-1]-np.array([103.939,116.779,123.68],dtype=np.float32))


def prepare_cv(tf):
    class_index=read_json(ROOT/'config/imagenet_class_index.json')
    wnid_to_id={value[0]:int(key) for key,value in class_index.items()}
    records={}
    tensors={}
    with tarfile.open(CACHE/'imagenette2-160.tgz','r:gz') as archive:
        members=[m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith('.jpeg')]
        for split,source_split in (('calibration','train'),('evaluation','val')):
            eligible=sorted([m for m in members if Path(m.name).parts[1]==source_split],key=lambda m:m.name)
            labels=[wnid_to_id[Path(m.name).parts[2]] for m in eligible]
            selected=balanced_selection(eligible,labels,10)
            prepared={}
            # Read gzip members in archive order to avoid decompressing from the
            # start on every random image. Restore seeded sample order afterward.
            for member in sorted(selected,key=lambda m:m.offset_data):
                source=archive.extractfile(member).read()
                filename=Path(member.name).name
                target=DATA/'cv/images'/source_split/Path(member.name).parts[2]/filename
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_bytes(source)
                x=cv_preprocess(target)
                label=wnid_to_id[Path(member.name).parts[2]]
                row={'source_path':member.name,'image_path':str(target.relative_to(ROOT)),
                             'label':label,'class_name':class_index[str(label)][1],
                             'source_sha256':hashlib.sha256(source).hexdigest(),
                             'tensor_sha256':hashlib.sha256(x.tobytes()).hexdigest()}
                prepared[member.name]=(x,label,row)
            xs,ys,rows=zip(*(prepared[m.name] for m in selected))
            records[split]=list(rows)
            tensors[split]={'image':np.stack(xs),'labels':np.asarray(ys,dtype=np.int64)}
    if {x['source_sha256'] for x in records['calibration']} & {x['source_sha256'] for x in records['evaluation']}:
        raise ValueError('CV split leakage: duplicate image bytes')
    model=tf.keras.applications.ResNet50(weights=str(CACHE/'resnet50.h5'),include_top=True,
                                        classifier_activation='softmax')
    original=ARTIFACTS/'cv/original.keras'
    original.parent.mkdir(parents=True,exist_ok=True)
    model.save(original)
    write_json(ARTIFACTS/'cv/model_info.json',{'model':'Keras ResNet50','parameters':model.count_params(),
        'original_sha256':sha256(original),'weights_sha256':sha256(CACHE/'resnet50.h5'),
        'output_kind':'softmax probabilities','input_layout':'NHWC','output_shape':[1,1000]})
    save_dataset('cv',records,tensors,{'dataset':'Imagenette 160px','seed':SEED,
        'preprocess':'EXIF -> RGB -> bilinear resize 224x224 -> BGR -> subtract [103.939,116.779,123.68]',
        'label_mapping':'Imagenette synset -> original ImageNet 1000-class index'})


def prepare_text(tf):
    import pyarrow.parquet as parquet
    from transformers import AutoTokenizer, TFDistilBertForSequenceClassification
    tokenizer=AutoTokenizer.from_pretrained(CACHE/'text_model',local_files_only=True)
    raw={}
    for split,name in (('calibration','train'),('evaluation','validation')):
        rows=parquet.read_table(CACHE/'sst2/data'/f'{name}-00000-of-00001.parquet').to_pylist()
        raw[split]=sorted(rows,key=lambda r:r['idx'])
    all_validation={normalize_text(r['sentence']) for r in raw['evaluation']}
    raw['calibration']=[r for r in raw['calibration'] if normalize_text(r['sentence']) not in all_validation]
    records,tensors={},{}
    for split,eligible in raw.items():
        selected=balanced_selection(eligible,[r['label'] for r in eligible],50)
        # Exact duplicate sentences must not be counted twice within a split.
        if len({normalize_text(r['sentence']) for r in selected})!=100:
            raise ValueError('Duplicate text within selected split')
        encoded=tokenizer([r['sentence'] for r in selected],padding='max_length',truncation=True,
                          max_length=SEQ,return_tensors='np')
        rows=[]
        for index,record in enumerate(selected):
            row={**record,'source_split':'train' if split=='calibration' else 'validation',
                 'untruncated_tokens':len(tokenizer(record['sentence'],truncation=False)['input_ids']),
                 'tensor_sha256':hashlib.sha256(encoded['input_ids'][index].astype(np.int32).tobytes()+
                                                 encoded['attention_mask'][index].astype(np.int32).tobytes()).hexdigest()}
            rows.append(row)
        records[split]=rows
        tensors[split]={k:np.asarray(encoded[k],dtype=np.int32) for k in ('input_ids','attention_mask')}
        tensors[split]['labels']=np.array([r['label'] for r in selected],dtype=np.int64)
    model=TFDistilBertForSequenceClassification.from_pretrained(CACHE/'text_model',local_files_only=True)
    assert model.config.id2label=={0:'NEGATIVE',1:'POSITIVE'}
    write_json(ARTIFACTS/'text/model_info.json',{'model':'DistilBERT SST-2','parameters':model.count_params(),
        'original_sha256':sha256(CACHE/'text_model/tf_model.h5'),'output_kind':'logits',
        'output_shape':[1,2],'sequence_length':SEQ,'id2label':model.config.id2label,
        'revision':read_json(ROOT/'sources.lock.json')['text_model']['revision']})
    save_dataset('text',records,tensors,{'dataset':'SST-2','seed':SEED,'sequence_length':SEQ,
        'preprocess':'original uncased WordPiece tokenizer; right padding/truncation to 64 tokens',
        'label_mapping':{'0':'NEGATIVE','1':'POSITIVE'},
        'truncated_evaluation':sum(r['untruncated_tokens']>SEQ for r in records['evaluation'])})


def save_dataset(task,records,tensors,metadata):
    folder=DATA/task
    folder.mkdir(parents=True,exist_ok=True)
    hashes={}
    for split,data in tensors.items():
        path=folder/(split+'.npz')
        np.savez_compressed(path,**data)
        hashes[split]=sha256(path)
    write_json(folder/'manifest.json',{'task':task,**metadata,'records':records,'tensor_hashes':hashes})
    print(f'{task}: 100 calibration + 100 labelled evaluation samples prepared',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,choices=('cv','text'))
    args=parser.parse_args()
    tf=tensorflow()
    (prepare_cv if args.task=='cv' else prepare_text)(tf)


if __name__=='__main__': main()
