"""Build the Vietnamese report, CSV and offline HTML from measured results."""
import csv
import html
import json
from datetime import datetime, timedelta, timezone

from .common import ARTIFACTS, DATA, RESULTS, ROOT, VARIANTS, model_path, read_json, sha256, write_json

TITLES={'cv':'ResNet50 — phân loại ảnh','text':'DistilBERT — sentiment tiếng Anh'}
LABELS={'tensorflow_fp32':'TensorFlow FP32','onnx_fp32':'ONNX FP32',
        'onnx_dynamic':'ONNX dynamic INT8','onnx_static':'ONNX static INT8 QDQ',
        'tflite_fp32':'TFLite FP32','tflite_dynamic':'TFLite dynamic range',
        'tflite_fp16':'TFLite FP16 weights','tflite_static':'TFLite static INT8'}


def number(value,places=3):
    if value is None: return '—'
    if isinstance(value,str): return value
    return f'{value:.{places}f}'


def collect():
    output={}
    for task in ('cv','text'):
        summary=read_json(RESULTS/task/'evaluation_summary.json')
        manifest=read_json(DATA/task/'manifest.json')
        info=read_json(ARTIFACTS/task/'model_info.json')
        rows=[]
        for variant in VARIANTS:
            evaluation=summary['results'][variant]
            conversion=None if variant=='tensorflow_fp32' else read_json(RESULTS/task/f'conversion_{variant}.json')
            row={'variant':variant,'label':LABELS[variant],**evaluation}
            if task=='text' and variant=='tflite_static': row['label']='TFLite calibrated mixed INT8'
            if evaluation['status']=='ok':
                row['benchmark']=read_json(RESULTS/task/f'benchmark_{variant}.json')
                row['size_mib']=(read_json(RESULTS/task/f'evaluation_{variant}.json')['model_size_bytes'] if variant=='tensorflow_fp32'
                                  else conversion['size_bytes'])/1048576
                row['conversion']=conversion
            else: row['conversion']=conversion
            rows.append(row)
        bases={r['variant']:r for r in rows}
        for row in rows:
            if row['status']=='ok':
                base=bases['onnx_fp32' if row['variant'].startswith('onnx') else 'tflite_fp32'
                           if row['variant'].startswith('tflite') else 'tensorflow_fp32']
                row['size_reduction_x']=base['size_mib']/row['size_mib']
                row['median_speedup_x']=base['benchmark']['median_ms']/row['benchmark']['median_ms']
                row['accuracy_delta_pp']=100*(row['metrics']['accuracy']-bases['tensorflow_fp32']['metrics']['accuracy'])
        output[task]={'title':TITLES[task],'model':info,'summary':summary,'manifest':manifest,'rows':rows}
    return output


def main():
    data=collect()
    measured=datetime.now(timezone(timedelta(hours=7))).isoformat()
    write_json(RESULTS/'summary.json',{'report_date':'2026-10-08','generated_at_utc7':measured,'tasks':data})
    lines=['# Báo cáo quantization: mô hình ảnh và text','', '**Ngày công việc: 08/10/2026.**','',
        '## 1. Mục tiêu công việc','',
        'Tìm hiểu và thực nghiệm post-training quantization bằng TFLite và ONNX Runtime. Dùng hai pretrained model khác bài MobileNetV2: ResNet50 cho CV và DistilBERT fine-tune SST-2 cho NLP. Đánh giá độ sai khác output, accuracy/F1, dung lượng và latency trên CPU máy tính; đóng ZIP kèm code và bằng chứng.','',
        '## 2. Kiến thức đã tìm hiểu','',
        '- Quantization biểu diễn giá trị thực bằng số nguyên: `q = clip(round(r/scale)+zero_point)`, `r_approx = scale*(q-zero_point)`. Phải clip trước cast để tránh wraparound INT8.',
        '- Static PTQ lấy scale/zero-point từ calibration set; dynamic quantization tính tham số activation trong lúc inference cho operator được hỗ trợ. ONNX Runtime khuyến nghị static cho CNN và dynamic cho Transformer/RNN.',
        '- TFLite dynamic range quantize weights; FP16 chủ yếu giảm storage, CPU có thể dequantize weights về FP32. Full INT8 cần representative dataset và kernels phù hợp.',
        '- QDQ chèn QuantizeLinear/DequantizeLinear vào graph ONNX; việc fusion/chạy kernel INT8 phụ thuộc runtime và provider. Không mặc định mọi operator trong model đã chuyển INT8.',
        '- Token ID/attention mask là chỉ số INT32, không phải activation FP32 để đổi trực tiếp thành INT8. Transformer có thể còn operator float; báo mixed INT8 nếu đúng graph thực tế.','',
        'Nguồn: [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [TFLite PTQ](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization).','',
        '## 3. Model, dữ liệu và điều kiện thử nghiệm','']
    for task,item in data.items():
        manifest=item['manifest'];model=item['model']
        timing=item['rows'][0]['benchmark']
        lines += [f'### {item["title"]}','',
            f"- Pretrained model: {model['model']}; {model['parameters']:,} parameters; không train lại.",
            f"- Dataset: {manifest['dataset']}; 100 calibration từ train, 100 evaluation từ val/validation, seed {manifest['seed']}. Hai tập không giao nhau theo hash ảnh/nội dung câu.",
            f"- Preprocess: {manifest['preprocess']}.",
            f"- Output để so số học: {item['summary']['domain']}. Dùng cùng canonical input giữa các runtime.",
            f"- CPU, batch=1, intra-op={timing['threads']}, inter-op=1 nếu API cho phép; {timing['warmup']} warm-up và {timing['runs']} lượt đo. Mỗi biến thể ở process riêng, chạy nối tiếp.",
            f"- Timer: {timing['timing_scope']}.",
            f"- Máy: {timing['system']['platform']}; CPU: {timing['system'].get('cpu_name',timing['system']['processor'])}; Python {timing['system']['python']}.",
            '- Phiên bản: TensorFlow 2.15.1, tf2onnx 1.16.1, ONNX Runtime 1.20.1; NLP thêm Transformers 4.38.2.', '']
        if task=='text': lines += [f"- Input token INT32 [1,64]; truncate {manifest['truncated_evaluation']}/100 câu evaluation. Nhãn 0=negative, 1=positive; chỉ tiếng Anh.", '']
        else: lines += ['- ResNet50 dùng BGR trừ mean ImageNet; không dùng normalization [-1,1] của MobileNetV2. Accuracy dự đoán trên đủ 1.000 lớp, không mask về 10 lớp Imagenette.','']
    lines+=['## 4. Sai khác output và accuracy','',
        'So với TensorFlow FP32 của chính model đó. Accuracy dùng nhãn thật; agreement là tỷ lệ nhãn dự đoán trùng model gốc. SNR = `10*log10(sum(reference²)/sum(error²))`, cộng trên toàn bộ output của 100 mẫu.','']
    csv_rows=[]
    for task,item in data.items():
        secondary='Top-5 accuracy' if task=='cv' else 'F1 positive'
        lines += [f'### {item["title"]}','',
            f'| Biến thể | MAE | RMSE | Max abs | SNR dB | Cosine | Accuracy | {secondary} | Agreement |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in item['rows']:
            if row['status']!='ok':
                lines.append(f"| {row['label']} | failed | — | — | — | — | — | — | — |")
                continue
            m=row['metrics'];b=row['benchmark']
            score=m.get('top5_accuracy',m.get('f1_positive'))
            lines.append(f"| {row['label']} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['max_abs_error']:.6g} | {number(m['snr_db'])} | {number(m['cosine_similarity'],12)} | {m['accuracy']*100:.1f}% | {score*100:.1f}% | {m['top1_agreement']*100:.1f}% |")
            csv_rows.append({'task':task,'variant':row['variant'],**m,'size_mib':row['size_mib'],
                'mean_ms':b['mean_ms'],'median_ms':b['median_ms'],'p95_ms':b['p95_ms'],
                'size_reduction_x':row['size_reduction_x'],'median_speedup_x':row['median_speedup_x'],
                'accuracy_delta_pp':row['accuracy_delta_pp']})
        lines+=['','Các bản FP32 phải đạt allclose trước khi diễn giải quantization. Không dùng tolerance FP32 làm tiêu chí thất bại cho INT8. Cosine làm tròn trong bảng; số đầy đủ nằm trong JSON/CSV.','']
    lines+=['## 5. Dung lượng model và tốc độ','',
        'Tỷ số dung lượng và speedup tính với FP32 **cùng runtime/cùng model**; speedup = median_FP32 / median_variant. Dung lượng file không phải RAM.','']
    for task,item in data.items():
        lines += [f'### {item["title"]}','',
            '| Biến thể | MiB | Giảm size × | Mean ms | Median ms | P95 ms | Speedup × |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in item['rows']:
            if row['status']!='ok':
                lines.append(f"| {row['label']} | failed | — | — | — | — | — |")
            else:
                b=row['benchmark']
                lines.append(f"| {row['label']} | {row['size_mib']:.2f} | {row['size_reduction_x']:.2f} | {b['mean_ms']:.3f} | {b['median_ms']:.3f} | {b['p95_ms']:.3f} | {row['median_speedup_x']:.2f} |")
        lines+=['']
    lines+=['## 6. Kết luận và giới hạn','']
    for task,item in data.items():
        rows=[r for r in item['rows'] if r['status']=='ok' and r['variant'] not in ('tensorflow_fp32','onnx_fp32','tflite_fp32')]
        best=min(rows,key=lambda r:r['benchmark']['median_ms'])
        lines += [f"- {item['title']}: bản quantized có median thấp nhất trong lượt đo là {best['label']}, {best['benchmark']['median_ms']:.3f} ms; accuracy {best['metrics']['accuracy']*100:.1f}%, chênh {best['accuracy_delta_pp']:+.1f} điểm phần trăm với TensorFlow gốc. Tốc độ này cần được cân nhắc cùng chất lượng."]
        for row in item['rows']:
            if row['status']!='ok':
                lines += [f"- {task}/{row['label']} không hoàn tất: {row['conversion'].get('error_type','unknown')}; xem `results/{task}/conversion_{row['variant']}.json` và log. Không có số đo được gán cho biến thể này."]
    text_rows={r['variant']:r for r in data['text']['rows']}
    bad=text_rows['tflite_static'];original=text_rows['tensorflow_fp32'];fp16=text_rows['tflite_fp16']
    if bad['status']=='ok':
        lines += [f"- **Không chọn TFLite mixed INT8 cho text trong cấu hình này:** accuracy {bad['metrics']['accuracy']*100:.1f}% so với {original['metrics']['accuracy']*100:.1f}% của model gốc ({bad['accuracy_delta_pp']:+.1f} điểm phần trăm), SNR {number(bad['metrics']['snr_db'])} dB. Convert thành công và chạy nhanh hơn không có nghĩa model đủ chất lượng."]
    if fp16['status']=='ok':
        lines += [f"- Text TFLite FP16 weights là lựa chọn đáng thử tiếp nếu cần giảm storage: {fp16['size_mib']:.2f} MiB, accuracy {fp16['metrics']['accuracy']*100:.1f}%, SNR {number(fp16['metrics']['snr_db'])} dB, speedup {fp16['median_speedup_x']:.2f}× so với TFLite FP32. ONNX FP32 vẫn có latency thấp hơn các bản ONNX quantized ở lượt đo này."]
    diagnostic=RESULTS/'text/diagnostic_tflite_static.json'
    if diagnostic.is_file():
        inspected=read_json(diagnostic)
        if inspected['model_sha256']!=text_rows['tflite_static']['conversion']['sha256']:
            raise ValueError('Diagnostic belongs to a different TFLite model')
        largest=inspected['largest_scales'][0]
        lines += [f"- Kiểm tra FlatBuffer text static thấy {inspected['scales_above_one_million']} tensor có scale > 10⁶, lớn nhất {largest['max_scale']:.4g}, xuất hiện quanh attention mask (`attention/mul_1`, `sub`, `add`, `MatMul`). Dải mask âm rất lớn có thể làm bước lượng tử quá thô và mất thông tin attention; đây là **suy luận chẩn đoán từ metadata**, chưa chứng minh nguyên nhân duy nhất bằng thí nghiệm loại trừ. Bằng chứng: `results/text/diagnostic_tflite_static.json`; chạy lại `python -m qlab.diagnose` sau khi tạo model."]
    lines += ['- Audit dtype/operator được lưu trong conversion JSON. Với ONNX dynamic ResNet50 chỉ chọn MatMul/Gemm, phần Conv có thể vẫn FP32 nên không kỳ vọng giảm file 4×. NLP static TFLite được gọi mixed INT8, không full INT8.',
        '- Giảm dung lượng không bảo đảm giảm latency: dynamic INT8 của ONNX text và dynamic range của TFLite CV chậm hơn FP32 cùng runtime ở lượt đo này. Nguyên nhân có thể liên quan kernel, overhead hoặc phần graph còn float; chưa có profiling để xác định tỷ lệ đóng góp.',
        '- Benchmark chỉ một phiên trên máy dùng chung, không khóa power mode/nhiệt độ/tác vụ nền. Một số P95 cách xa median (đặc biệt ONNX FP32 CV); xem đủ 200 mẫu và độ lệch chuẩn. Speedup là mô tả phiên đo, chưa chứng minh mức cải thiện ổn định qua nhiều phiên.',
        '- INT8 softmax có độ phân giải hữu hạn: không renormalize output để giấu sai khác. JSON ghi tổng xác suất và số tie top-1.',
        '- Chỉ 100 ảnh Imagenette 160px resized và 100 câu SST-2 validation; kết quả không phải accuracy toàn bộ ImageNet/SST-2. Chênh +1 điểm phần trăm chỉ là thêm 1 mẫu đúng, chưa chứng minh quantization cải thiện chất lượng tổng quát. Một ảnh/một câu fixed length được lặp để benchmark, không đại diện mọi input.',
        '- Chưa chạy các model quantized này trên Android, GPU, NNAPI; kết quả PC không đại diện điện thoại. Chưa đo RAM, năng lượng hoặc hiệu quả QAT.',
        '- Calibration và lựa chọn op-types cố định trước evaluation. Không dùng evaluation để chọn scale hay tuning lại model.', '',
        '## 7. Hướng thực hiện tiếp theo','',
        'Đo thêm nhiều mẫu/lượt, khảo sát độ dài text 32/128 token thành thử nghiệm riêng, kiểm tra calibration đa dạng hơn và triển khai các biến thể lên Android thật. Nếu accuracy giảm quá mức, cân nhắc calibration khác hoặc QAT; các bước này chưa thực hiện.','',
        '## 8. Code và bằng chứng bàn giao','',
        '- `README.md`: cách chạy và giải thích từng module; `hoc-quantization.html`: học liệu offline tương tác.',
        '- `run_lab.py`, `qlab/`, `tests/`, `requirements.txt`, `requirements-lock.txt`, `sources.lock.json`.',
        '- `results/summary.json`, `results/summary.csv`, `results/cv/outputs.npz`, `results/text/outputs.npz`: full raw outputs, nhãn và metric.',
        '- `results/<task>/benchmark_*.json`: đủ mẫu latency; `conversion_*.json`: hash, strategy và audit; `data/<task>/manifest.json`: input/nhãn/source hashes.',
        '- ZIP đóng theo allowlist, không chứa môi trường ảo, cache, credentials hay model/dataset lớn. Lệnh chạy lại tự tải từ revision/checksum đã pin.',
        '- Unit tests và verifier kiểm tra phép quantization, label metrics, split leakage, hash output và thống kê latency. Verifier chạy được từ ZIP giải nén, không cần tải model.', '',
        f'Thời điểm tổng hợp (UTC+7): {measured}.']
    (ROOT/'bao-cao-quantization-2026-10-08.md').write_text('\n'.join(lines),encoding='utf-8')
    fields=list(dict.fromkeys(k for row in csv_rows for k in row))
    with (RESULTS/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(csv_rows)
    sources={p.relative_to(ROOT).as_posix():p.read_text(encoding='utf-8') for p in sorted((ROOT/'qlab').glob('*.py'))}
    sources['run_lab.py']=(ROOT/'run_lab.py').read_text(encoding='utf-8')
    template=(ROOT/'learning.template.html').read_text(encoding='utf-8')
    packed=json.dumps({'tasks':data,'sources':sources},ensure_ascii=False).replace('</','<\\/')
    (ROOT/'hoc-quantization.html').write_text(template.replace('__LAB_DATA__',packed),encoding='utf-8')
    print('Report, CSV, JSON and HTML generated from measured outputs and latency.',flush=True)


if __name__=='__main__': main()
