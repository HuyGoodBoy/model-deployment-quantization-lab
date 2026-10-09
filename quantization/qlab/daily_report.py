"""Full Vietnamese daily Markdown and a concise, measured chat report."""
import csv
from datetime import datetime, timedelta, timezone

import numpy as np

from .common import DATA, RESULTS, ROOT, numerical_metrics, read_json, write_json
from .gpu_runtime import GPU_VARIANTS
from .report import collect, LABELS, TITLES

DATE = '2026-10-08'
FULL_REPORT = ROOT / f'bao-cao-ngay-{DATE}-cpu-gpu.md'
CHAT_REPORT = ROOT / f'bao-cao-chat-{DATE}.md'
GPU_LABELS = {**LABELS, 'onnx_fp16': 'ONNX FP16 (IO giữ nguyên)'}


def fmt(value, digits=3):
    if value is None:
        return '—'
    if value == 'inf':
        return '∞'
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) else str(value)


def pct(value):
    return f'{100*value:.1f}%'


def gpu_data():
    data = {}
    for task in ('cv', 'text'):
        rows = []
        outputs = {}
        folder = RESULTS / 'gpu' / task
        for variant in GPU_VARIANTS:
            for device in ('cpu', 'cuda'):
                path = folder / f'{device}_{variant}.json'
                row = read_json(path) if path.exists() else {'status': 'not_run', 'task': task,
                                                            'variant': variant, 'device': device}
                rows.append(row)
                if row['status'] == 'ok':
                    outputs[(device, variant)] = np.load(folder / f'output_{device}_{variant}.npy',
                                                        allow_pickle=False)
        lookup = {(r['device'], r['variant']): r for r in rows}
        for row in rows:
            if row['status'] != 'ok':
                continue
            cpu = lookup[('cpu', row['variant'])]
            if row['device'] == 'cuda' and cpu['status'] == 'ok':
                row['speedup_vs_paired_cpu'] = cpu['benchmark']['median_ms']/row['benchmark']['median_ms']
                a, b = outputs[('cpu', row['variant'])], outputs[('cuda', row['variant'])]
                row['metrics_vs_paired_cpu'] = {**numerical_metrics(a, b),
                    'top1_agreement': float(np.mean(a.argmax(1) == b.argmax(1)))}
            if row['device'] == 'cuda' and ('cuda', 'onnx_fp32') in outputs:
                row['metrics_vs_cuda_fp32'] = numerical_metrics(outputs['cuda', 'onnx_fp32'],
                                                               outputs['cuda', row['variant']])
            if row['device'] == 'cuda':
                events = read_json(ROOT / row['profile']['file'])
                types = {}
                seen = set()
                for event in events:
                    args = event.get('args', {})
                    op = args.get('op_name')
                    key = event.get('name', '')
                    if (args.get('provider') != 'CUDAExecutionProvider' or key in seen
                            or not key.endswith('_kernel_time') or op not in ('Conv','MatMul','Gemm','GlobalAveragePool')):
                        continue
                    seen.add(key)
                    dtype = sorted({dtype for value in args.get('input_type_shape', [])
                                    if isinstance(value, dict) for dtype in value})
                    types.setdefault(op, set()).update(dtype)
                row['cuda_heavy_op_input_types'] = {op: sorted(value) for op,value in types.items()}
        data[task] = rows
    return data


def main():
    cpu = collect()
    gpu = gpu_data()
    environment = read_json(RESULTS / 'gpu/environment.json')
    generated = datetime.now(timezone(timedelta(hours=7))).isoformat()
    repeat_folder = RESULTS / 'gpu/cv/first_pass_fp32'
    initial_pair = ({device: read_json(repeat_folder/f'{device}_onnx_fp32.json')
                     for device in ('cpu','cuda')} if repeat_folder.exists() else {})
    write_json(RESULTS / 'gpu/summary.json', {'report_date': DATE, 'generated_at_utc7': generated,
               'environment': environment, 'tasks': gpu, 'resnet_fp32_initial_pair': initial_pair})
    csv_rows = []
    for task, rows in gpu.items():
        for row in rows:
            flat = {'phase': 'primary', 'task': task, 'variant': row['variant'], 'device': row['device'], 'status': row['status']}
            if row['status'] == 'ok':
                flat.update(row['metrics_vs_original'])
                flat.update({k: row['benchmark'][k] for k in ('mean_ms', 'median_ms', 'p95_ms', 'std_ms')})
                flat.update({'size_mib': row['model_size_bytes']/1048576,
                             'speedup_vs_paired_cpu': row.get('speedup_vs_paired_cpu'),
                             'cuda_compute_nodes': row['profile']['cuda_compute_nodes'],
                             'cpu_nodes': row['profile']['unique_nodes_by_provider'].get('CPUExecutionProvider', 0)})
            csv_rows.append(flat)
    for device,row in initial_pair.items():
        csv_rows.append({'phase':'first_pass_resnet_fp32','task':'cv','variant':'onnx_fp32',
                         'device':device,'status':row['status'],**row['metrics_vs_original'],
                         **{key:row['benchmark'][key] for key in ('mean_ms','median_ms','p95_ms','std_ms')}})
    with (RESULTS / 'gpu/summary.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in csv_rows for k in r)))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ['# Báo cáo thực nghiệm quantization ResNet50 và DistilBERT — 08/10/2026', '',
             '## 1. Công việc và kết quả chính', '',
             '- Thực nghiệm **ResNet50 pretrained ImageNet** và **DistilBERT fine-tune SST-2 tiếng Anh**, không train lại.',
             '- Đã chạy 8 biến thể/model trên CPU với ONNX Runtime và TFLite; bổ sung ONNX FP32, dynamic INT8, static QDQ INT8 và FP16 trên CPU/CUDA cùng phiên bản runtime.',
             '- Đánh giá 100 ảnh + 100 câu có nhãn; calibration riêng 100 mẫu/model. Đo cả sai khác output, accuracy/F1, dung lượng và latency.',
             '- **Text TFLite mixed INT8 chưa đạt chất lượng:** accuracy 52% so với 91% của model gốc; lưu nguyên kết quả và bằng chứng chẩn đoán attention scale.',
             '- GPU được xác minh bằng profiling provider/operator. Có CUDA trong danh sách provider chưa đủ để khẳng định graph chạy GPU.', '',
             '## 2. Model, dữ liệu và cách đánh giá', '',
             '| Nội dung | CV | Text/NLP |', '|---|---|---|',
             '| Model | Keras ResNet50, 1.000 lớp ImageNet | DistilBERT SST-2, 2 nhãn cảm xúc tiếng Anh |',
             '| Calibration | 100 ảnh train Imagenette, 10/lớp | 100 câu SST-2 train, 50/nhãn |',
             '| Evaluation | 100 ảnh val Imagenette, 10/lớp | 100 câu SST-2 validation, 50/nhãn |',
             '| Input | FP32 NHWC [1,224,224,3] | INT32 input_ids + attention_mask [1,64] |',
             '| Preprocess | RGB resize bilinear, BGR trừ mean ResNet | WordPiece uncased gốc, pad/truncate 64 |',
             '| Output so sánh | 1.000 softmax probabilities | 2 logits |',
             '| Chất lượng | Accuracy/top-5 trên đủ 1.000 lớp | Accuracy/F1 positive, nhãn 0 negative/1 positive |', '',
             f"- Seed 20261008; kiểm tra hai tập không giao theo hash ảnh/nội dung câu. Text có {cpu['text']['manifest']['truncated_evaluation']}/100 câu evaluation bị truncate.",
             '- Tất cả runtime dùng cùng canonical input đã kiểm checksum. Token ID/mask giữ INT32, không đổi thành INT8.',
             '- Model gốc để so output: TensorFlow FP32 của chính model đó. Accuracy cần nhãn thật; agreement chỉ đo nhãn dự đoán trùng model gốc.',
             '- MAE = mean(|error|), RMSE = sqrt(mean(error²)), max = max(|error|); SNR = 10 log₁₀(Σreference²/Σerror²). Cosine gần 1 đo hướng vector, không bảo đảm độ lớn giống nhau.',
             '- SNR tính gộp toàn output của 100 mẫu; không so trực tiếp SNR CV probabilities với NLP logits. JSON/CSV giữ thêm relative L2 và số liệu đầy đủ.', '',
             '## 3. Môi trường và giao thức đo', '',
             f"- CPU: **{environment['system']['cpu_name']}**, RAM máy 16 GB; Windows x64, Python {environment['system']['python']}.",
             '- GPU thực tế: **NVIDIA GeForce RTX 3050 Laptop GPU, 4.096 MiB VRAM, compute capability 8.6**, NVIDIA driver **572.61**.',
             '- Phiên CPU ban đầu: TensorFlow 2.15.1, ONNX Runtime 1.20.1, TFLite Interpreter đi kèm TensorFlow 2.15.1; Transformers 4.38.2.',
             f"- Phiên CPU/GPU bổ sung: **onnxruntime-gpu {environment['runtime_version']}**; cả hai provider dùng cùng wheel/runtime và cùng model hash. GPU venv riêng, không sửa môi trường CPU.",
             f"- CUDA runtime {environment['package_versions']['nvidia-cuda-runtime-cu12']}, cuBLAS {environment['package_versions']['nvidia-cublas-cu12']}, cuDNN {environment['package_versions']['nvidia-cudnn-cu12']}; DLL từ các wheel NVIDIA cài trong venv.",
             '- `nvidia-smi` báo CUDA 12.8 là khả năng của driver, không chứng minh đã cài CUDA Toolkit 12.8. Thư viện runtime thực tế được ghi riêng ở trên.',
             '- Batch 1; intra-op CPU 1, inter-op 1; **30 warm-up + 200 lượt/biến thể**, process riêng chạy nối tiếp. Cùng ảnh/câu đầu tiên được lặp để đo.',
             '- GPU dùng **session.run đồng bộ với NumPy input/output trên CPU**: gồm H2D/D2H, API và overhead Python. Loại model load, đọc file, preprocess/tokenization; không phải kernel-only.',
             '- CUDA EP đặt `use_tf32=0`, `device_id=0`, copy trong default stream; các option thực tế được lưu trong JSON. Profiling bằng session riêng sau timing, không làm tăng latency benchmark.',
             '- Speedup GPU = median CPU phiên bổ sung / median CUDA của **cùng variant**. Không lấy CPU 1.20.1 phiên trước làm mẫu số cho GPU 1.20.2.',
             '- TensorFlow cài sẵn báo `is_built_with_cuda=False`, GPU physical list rỗng. TFLite Interpreter 2.15 trong bài chưa cấu hình GPU delegate; các hàng TFLite dưới đây đều CPU. Không kết luận mọi LiteRT/Windows đều không hỗ trợ GPU.', '',
             '## 4. Các kiểu quantization đã thử', '',
             '- ONNX dynamic: QInt8 weights per-channel, chọn MatMul/Gemm. ResNet Conv và embedding có thể còn FP32, không gọi toàn bộ model là INT8.',
             '- ONNX static: QDQ S8S8, MinMax calibration 100 mẫu; CV chọn Conv/MatMul/Gemm, NLP chọn MatMul/Gemm. Runtime có thể fusion hoặc chia graph theo provider.',
             '- TFLite dynamic range: Optimize.DEFAULT không representative dataset. FP16 weights giảm storage, không bảo đảm CPU compute FP16.',
             '- TFLite static CV: builtins INT8, input/output INT8, audit không còn tensor FP32. Text cho phép mixed INT8/float, giữ token IDs/mask INT32; không bật SELECT_TF_OPS.',
             '- ONNX FP16 bổ sung: chuyển từ ONNX FP32, giữ kiểu IO, dùng default blocked operators; không phải integer PTQ. Converter clip constants về finite bounds 1e-7…1e4 nên sentinel attention mask có thể đổi; kiểm chất lượng bằng dữ liệu evaluation, không tune bằng evaluation.', '',
             '## 5. Kết quả CPU ban đầu — ONNX Runtime và TFLite', '',
             'Các số liệu dưới đây giữ nguyên phiên CPU đã thực hiện. Median/P95 tính từ đủ 200 mẫu; dung lượng file là MiB, không phải RAM.', '']
    for task, item in cpu.items():
        secondary = 'Top-5' if task == 'cv' else 'F1 positive'
        lines += [f'### {TITLES[task]}', '',
                  f'| Biến thể | MiB | Median ms | P95 ms | Accuracy | {secondary} | Max error | SNR dB |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in item['rows']:
            if row['status'] != 'ok':
                lines.append(f"| {row['label']} | failed | — | — | — | — | — | — |")
                continue
            m, b = row['metrics'], row['benchmark']
            score = m['top5_accuracy'] if task == 'cv' else m['f1_positive']
            lines.append(f"| {row['label']} | {fmt(row['size_mib'],2)} | {fmt(b['median_ms'])} | {fmt(b['p95_ms'])} | {pct(m['accuracy'])} | {pct(score)} | {m['max_abs_error']:.5g} | {fmt(m['snr_db'])} |")
        lines += ['', f"Bảng MAE/RMSE/cosine/agreement đầy đủ: [báo cáo CPU](bao-cao-quantization-{DATE}.md), [JSON](results/summary.json).", '']
    lines += ['## 6. Kết quả chạy GPU và CPU đối chứng mới', '',
              'Mỗi cặp dùng cùng graph/hash, ONNX Runtime 1.20.2 và cùng input. “CUDA + CPU” nghĩa profiling thấy graph có cả hai provider, kể cả khi chỉ một số op hỗ trợ/shape ở CPU. Không gọi INT8 CUDA là full INT8 GPU chỉ theo tên file.', '']
    if initial_pair:
        final = {(r['device'],r['variant']):r for r in gpu['cv']}
        lines += ['### Kiểm tra dao động ResNet FP32: giữ cả hai lượt', '',
                  'CPU FP32 lượt đầu lệch lớn với phiên CPU cũ và bản dynamic, nên đo thêm đúng một cặp CPU/CUDA sau các phép đo khác. Bảng chính dùng cặp đo cuối, không chọn minimum của hai lượt. Không có bằng chứng để quy chênh lệch riêng cho phiên bản runtime hoặc GPU.', '',
                  '| Lượt | CPU median / P95 ms | CUDA median / P95 ms | Speedup cùng lượt |',
                  '|---|---:|---:|---:|']
        for name,pair in [('Đầu tiên',initial_pair),('Đối chiếu cuối',{d:final[d,'onnx_fp32'] for d in ('cpu','cuda')})]:
            c,g = pair['cpu']['benchmark'],pair['cuda']['benchmark']
            lines.append(f"| {name} | {c['median_ms']:.3f} / {c['p95_ms']:.3f} | {g['median_ms']:.3f} / {g['p95_ms']:.3f} | {c['median_ms']/g['median_ms']:.2f}× |")
        lines += ['', 'Raw data lượt đầu giữ ở `results/gpu/cv/first_pass_fp32/`, cùng raw profile gốc. Hai phiên CPU cũ/mới còn khác môi trường/thời điểm; chưa có nhiều lượt kiểm soát để kết luận tốc độ ổn định.', '']
    for task, rows in gpu.items():
        lines += [f'### {TITLES[task]} — dung lượng và latency', '',
                  '| Biến thể | Thiết bị thực thi | MiB | Mean ms | Median ms | P95 ms | GPU/CPU speedup |',
                  '|---|---|---:|---:|---:|---:|---:|']
        for row in rows:
            label = GPU_LABELS[row['variant']]
            if row['status'] != 'ok':
                lines.append(f"| {label} | {row['device']}: {row['status']} | — | — | — | — | — |")
                continue
            provider = 'CPU' if row['device'] == 'cpu' else 'CUDA + CPU' if row['profile']['unique_nodes_by_provider'].get('CPUExecutionProvider') else 'CUDA'
            b = row['benchmark']
            lines.append(f"| {label} | {provider} | {fmt(row['model_size_bytes']/1048576,2)} | {fmt(b['mean_ms'])} | {fmt(b['median_ms'])} | {fmt(b['p95_ms'])} | {fmt(row.get('speedup_vs_paired_cpu'),2)}{'×' if row.get('speedup_vs_paired_cpu') else ''} |")
        secondary = 'Top-5' if task == 'cv' else 'F1 positive'
        lines += ['', f'### {TITLES[task]} — output CUDA so TensorFlow gốc', '',
                  f'| Biến thể | MAE | RMSE | Max error | SNR dB | Cosine | Accuracy | {secondary} | Agreement |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['device'] != 'cuda':
                continue
            label = GPU_LABELS[row['variant']]
            if row['status'] != 'ok':
                lines.append(f'| {label} | {row["status"]} | — | — | — | — | — | — | — |')
                continue
            m = row['metrics_vs_original']
            score = m['top5_accuracy'] if task == 'cv' else m['f1_positive']
            lines.append(f"| {label} | {m['mae']:.6g} | {m['rmse']:.6g} | {m['max_abs_error']:.6g} | {fmt(m['snr_db'])} | {fmt(m['cosine_similarity'],10)} | {pct(m['accuracy'])} | {pct(score)} | {pct(m['top1_agreement'])} |")
        lines += ['', f'### {TITLES[task]} — kiểm chứng provider và sai khác cùng graph', '',
                  '| Biến thể CUDA | Node CUDA / CPU | Op tính toán trên CUDA | Op trên CPU | Max error so cùng variant CPU |',
                  '|---|---|---|---|---:|']
        for row in rows:
            if row['device'] != 'cuda' or row['status'] != 'ok':
                continue
            p = row['profile']; counts = p['unique_nodes_by_provider']
            cpu_ops = p['operator_counts_by_provider'].get('CPUExecutionProvider', {})
            cpu_desc = ', '.join(f'{op}×{n}' for op, n in sorted(cpu_ops.items())) or '—'
            pair_error = row.get('metrics_vs_paired_cpu', {}).get('max_abs_error')
            error_text = '—' if pair_error is None else f'{pair_error:.6g}'
            lines.append(f"| {GPU_LABELS[row['variant']]} | {counts.get('CUDAExecutionProvider',0)} / {counts.get('CPUExecutionProvider',0)} | {', '.join(p['cuda_compute_operators'])} | {cpu_desc} | {error_text} |")
        lines += ['', 'Node count là số node duy nhất sau optimization, không nhân với số lượt profiling; không phải phần trăm FLOPs. Raw profile lưu type/shape của input kernel, các phép memcpy và CPU fallback. TensorRT/INT8 tensor core chưa được thử.', '']
        for row in rows:
            if row['device'] == 'cuda' and row['status'] == 'ok' and row['variant'] in ('onnx_static','onnx_fp16'):
                lines += [f"- Dtype input của op CUDA `{row['variant']}` đọc từ profile: `{row['cuda_heavy_op_input_types']}`. INT8 ở storage/QDQ không đồng nghĩa Conv/MatMul tính INT8; FP16 input cũng không tự chứng minh kiểu accumulation hoặc tensor core đã được sử dụng."]
        lines += ['']
        for row in rows:
            if row['status'] not in ('ok',):
                lines += [f"- {task}/{row['device']}/{row['variant']}: **{row['status']}**; {row.get('error',row.get('reason','xem JSON trạng thái/conversion và log'))}. Không gán số đo GPU thành công cho ca này."]
    lines += ['', '## 7. Nhận xét và lựa chọn tiếp theo', '']
    for task, rows in gpu.items():
        lookup = {(r['device'],r['variant']): r for r in rows}
        fp32, fp16 = lookup.get(('cuda','onnx_fp32')), lookup.get(('cuda','onnx_fp16'))
        if fp32 and fp32['status'] == 'ok':
            lines += [f"- **{TITLES[task]}:** ONNX FP32 CUDA median {fp32['benchmark']['median_ms']:.3f} ms, speedup {fp32['speedup_vs_paired_cpu']:.2f}× so với CPU cùng graph; accuracy {pct(fp32['metrics_vs_original']['accuracy'])}. FP32 đạt allclose so TensorFlow gốc."]
        if fp16 and fp16['status'] == 'ok':
            lines += [f"- ONNX FP16 CUDA {task}: median {fp16['benchmark']['median_ms']:.3f} ms, {fp16['model_size_bytes']/1048576:.2f} MiB, accuracy {pct(fp16['metrics_vs_original']['accuracy'])}, SNR {fmt(fp16['metrics_vs_original']['snr_db'])} dB. Precision và chất lượng phải đọc cùng tốc độ."]
        for row in rows:
            if row['device'] == 'cuda' and row['variant'] in ('onnx_dynamic','onnx_static') and row['status'] == 'ok':
                cpu_ops = row['profile']['operator_counts_by_provider'].get('CPUExecutionProvider', {})
                heavy = {op:n for op,n in cpu_ops.items() if op in ('MatMulInteger','QLinearConv','QLinearMatMul','MatMul','Gemm','Conv')}
                if heavy:
                    lines += [f"- {task}/{GPU_LABELS[row['variant']]}: profiling thấy các op tính toán {heavy} ở CPU. Đây là graph chạy kết hợp CPU/CUDA, không chứng minh kernel INT8 chính chạy trên GPU; transfer/fallback có thể ảnh hưởng latency."]
    diagnostic = read_json(RESULTS / 'text/diagnostic_tflite_static.json')
    lines += ['- **TFLite text mixed INT8 không chọn triển khai ở cấu hình này:** giảm 39 điểm phần trăm accuracy. FP16 weights giữ 91%, giảm file khoảng 2×; dynamic range giữ 90% nhưng không chắc nhanh hơn FP32.',
              '- ONNX FP16 CUDA của text đạt 89% (so với 91% FP32), chậm hơn FP32 CUDA ở lượt này. CPU cùng file FP16 vẫn đạt 91%; TFLite FP16 CPU cũng đạt 91%. Storage FP16, CUDA compute với FP16 và CPU có cast sang FP32 là các điều kiện khác nhau, phải đọc dtype/profile và metric thực tế; chưa xác định nguyên nhân số học duy nhất bằng thí nghiệm loại trừ.',
              '- QDQ static cùng file có accuracy khác giữa CPU/CUDA (CV 70%/73%, NLP 92%/91% ở lượt này). Profile CUDA thấy Conv/MatMul/Gemm nhận float; không gọi đây là full INT8 GPU hoặc khẳng định GPU cải thiện accuracy tổng quát.',
              '- ONNX CPU text dynamic/static phiên mới nhanh hơn FP32 CPU, trong khi phiên CPU ban đầu chậm hơn. Hai phiên khác thời điểm và wheel/runtime; chưa cô lập nguyên nhân. Giữ cả hai bảng thay vì suy rộng một phiên ra mọi cấu hình.',
              f"- Kiểm tra TFLite text static thấy {diagnostic['scales_above_one_million']} tensor scale > 10⁶, lớn nhất {diagnostic['largest_scales'][0]['max_scale']:.4g} quanh attention mask. Dải mask rất lớn có thể gây bước quantization quá thô; đây là suy luận từ metadata, chưa chứng minh nguyên nhân duy nhất bằng thí nghiệm loại trừ.",
              '- ONNX static CV CPU ban đầu giảm file khoảng 3,91× và median khoảng 1,42×, nhưng accuracy 70% so với 73% gốc: cần cân đối mức mất chất lượng. TFLite static CV đạt 74% trên subset, không suy ra cải thiện toàn dataset.',
              '- Giảm file không bảo đảm nhanh hơn. FP16 floating-point và INT8 integer có kernel/phạm vi hỗ trợ khác nhau; chọn theo graph, hardware và số đo thực tế.', '',
              '## 8. Giới hạn và việc chưa làm', '',
              '- Chỉ 100 mẫu/model; chênh 1 điểm phần trăm = 1 mẫu đúng. Không phải accuracy toàn ImageNet/SST-2, không đánh giá text tiếng Việt.',
              '- Một phiên/laptop Windows WDDM dùng chung; chưa khóa nhiệt độ/power mode/tác vụ nền hoặc chạy nhiều phiên để có confidence interval. CPU 1 thread không đại diện CPU tối ưu toàn bộ core.',
              '- Một input batch 1 lặp để timing, text cố định 64 token. Không đại diện throughput nhiều batch, mọi độ dài câu hay startup latency.',
              '- Benchmark H2D/D2H, chưa tối ưu I/O Binding/CUDA Graphs, chưa TensorRT, QAT hoặc tuning calibration. Không chỉnh quantization dựa trên evaluation.',
              '- Model quantized của bài này chưa chạy Android/GPU điện thoại; RTX 3050 không đại diện điện thoại. Chưa đo peak RAM/VRAM hoặc năng lượng; nvidia-smi chỉ kiểm cấu hình, không phải phép đo bộ nhớ đỉnh.', '',
              '## 9. Code, bằng chứng và cách chạy lại', '',
              '- [README GPU](README-gpu.md): môi trường riêng và các lệnh chạy.',
              '- `run_gpu.py`, `qlab/gpu_runtime.py`, `gpu_experiment.py`, `gpu_verify.py`, `daily_report.py`: runner, kiểm CUDA/provider, phép đo, verifier và báo cáo.',
              '- `../environments/gpu/requirements.txt` và `../environments/gpu/requirements-lock.txt`: package GPU, CUDA/cuDNN thực tế.',
              '- `results/gpu/<task>/output_*.npy`: đủ 100 output/variant/provider; `cpu_*.json`, `cuda_*.json`: đủ latency samples, metric, hash, provider options.',
              '- `results/gpu/<task>/profile_*.json`: raw profiling; `results/gpu/summary.json`, `summary.csv`: bảng tổng hợp.',
              '- `results/<task>/outputs.npz`, manifest và conversion audit CPU được giữ nguyên làm chuẩn.',
              '- Repo: [GitHub](https://github.com/HuyGoodBoy/model-deployment-quantization-lab). Giữ code, báo cáo và raw evidence cần thiết; bỏ venv/cache/credentials/models lớn, log và file build/ZIP.',
              '- Trạng thái kiểm chứng: kết quả CPU đã được kiểm chứng offline và 10 unit test đã pass trong quá trình thực nghiệm. Verifier GPU đã có mã nguồn nhưng chưa được chạy xác nhận trong lần bàn giao này. Các lệnh dưới đây phục vụ tái lập.', '',
              '```powershell', '.\\.venv-gpu\\Scripts\\python.exe run_gpu.py',
              '.\\.venv-gpu\\Scripts\\python.exe -m qlab.gpu_verify',
              '.\\.venv-gpu\\Scripts\\python.exe -m qlab.daily_report', '```', '',
              'Tài liệu chính thức: [ORT CUDA EP](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [ORT FP16](https://onnxruntime.ai/docs/performance/model-optimizations/float16.html), [ORT quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html), [TensorFlow Windows](https://www.tensorflow.org/install/pip#windows-native), [LiteRT GPU](https://ai.google.dev/edge/litert/next/gpu).', '',
              f'Tạo báo cáo lúc (UTC+7): {generated}.']
    FULL_REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    chat = ['Anh ơi, em báo cáo ngày 08/10 ạ.', '',
            '- Em thử quantization ONNX Runtime/TFLite trên ResNet50 (ảnh) và DistilBERT (sentiment text tiếng Anh), không train lại.',
            '- Dùng 100 mẫu calibration + 100 mẫu evaluation có nhãn/model, tách riêng. So output với TensorFlow gốc bằng MAE, max error, SNR, cosine; đo accuracy/F1, size và latency.',
            '- CPU: ONNX static ResNet50 giảm file 3,91×, median 80,91 → 57,00 ms, accuracy 73% → 70%. TFLite FP16 text giảm file 2×, accuracy giữ 91%.']
    for task, rows in gpu.items():
        lookup = {(r['device'],r['variant']):r for r in rows}
        c, g, h = lookup.get(('cpu','onnx_fp32')), lookup.get(('cuda','onnx_fp32')), lookup.get(('cuda','onnx_fp16'))
        if c and g and c['status'] == g['status'] == 'ok':
            short = 'ResNet50' if task == 'cv' else 'DistilBERT'
            detail = f"- RTX 3050 4 GB, {short} ONNX FP32: CPU {c['benchmark']['median_ms']:.2f} → CUDA {g['benchmark']['median_ms']:.2f} ms ({g['speedup_vs_paired_cpu']:.2f}×); accuracy {pct(g['metrics_vs_original']['accuracy'])}, SNR {fmt(g['metrics_vs_original']['snr_db'],1)} dB."
            if h and h['status'] == 'ok':
                detail += f" FP16 CUDA {h['benchmark']['median_ms']:.2f} ms, accuracy {pct(h['metrics_vs_original']['accuracy'])}."
            chat.append(detail)
    chat += ['- Text TFLite mixed INT8 chỉ đạt 52% so với FP32 91%, chưa phù hợp triển khai. INT8 không mặc định nhanh hơn; CUDA có thể còn CPU fallback, đã lưu profiling để kiểm tra.',
             '- Latency: batch 1, CPU 1 thread, 30 warm-up, 200 lượt; GPU gồm transfer input/output. Kết quả PC/subset, chưa suy ra Android.',
             '- Code và báo cáo: https://github.com/HuyGoodBoy/model-deployment-quantization-lab']
    CHAT_REPORT.write_text('\n'.join(chat)+'\n', encoding='utf-8')
    print(f'Generated {FULL_REPORT.name} and {CHAT_REPORT.name}', flush=True)


if __name__ == '__main__':
    main()
