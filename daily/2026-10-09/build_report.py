"""Build mentor Markdown and local chat report from measured JSON, never invented rows."""
import csv
import datetime
import json
from pathlib import Path

DAY=Path(__file__).resolve().parent


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def records(task,pattern):
    return [read(p) for p in sorted((DAY/'experiments'/task/'results/run-01').glob(pattern))]
def fmt(value,digits=3):
    return f'{value:.{digits}f}' if isinstance(value,(int,float)) else str(value)
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |',
                      *['| '+' | '.join(str(x) for x in row)+' |' for row in rows]])


def main():
    runtime=[r for r in records('01-reproduce','*.json') if isinstance(r,dict) and 'metrics' in r]
    distil=records('02-distilbert-int8','measure_*.json')
    mlsd=records('03-mlsd','measure_*.json')
    runtime_table=table(['Model / variant','ORT/LiteRT cũ ms','ORT/LiteRT mới ms','Mới/cũ','Accuracy cũ→mới','SNR cũ→mới dB'],[
        [f"{task}/{variant}",fmt(a['benchmark']['median_ms']),fmt(b['benchmark']['median_ms']),
         fmt(b['benchmark']['median_ms']/a['benchmark']['median_ms']),
         f"{a['metrics']['accuracy']:.0%} → {b['metrics']['accuracy']:.0%}",
         f"{fmt(a['metrics']['snr_db'])} → {fmt(b['metrics']['snr_db'])}"]
        for task,variant in sorted({(r['task'],r['variant']) for r in runtime})
        for a in runtime if a['task']==task and a['variant']==variant and '_old_' in a['name']
        for b in runtime if b['task']==task and b['variant']==variant and '_new_' in b['name']])
    distil_table=table(['Biến thể','Accuracy','F1','SNR dB','MAE','Max error','Median ms','Scale lớn nhất'],[
        [r['variant'],f"{r['metrics']['accuracy']:.0%}",fmt(r['metrics']['f1_positive']),
         fmt(r['metrics']['snr_db']),fmt(r['metrics']['mae'],5),fmt(r['metrics']['max_abs_error'],5),
         fmt(r['benchmark']['median_ms']),f"{r['audit']['max_quant_scale']:.3g}"] for r in distil])
    mlsd_table=table(['Biến thể / runtime','MiB','Median ms','p95 ms','SNR vs Torch','Max raw error','Line demo'],[
        [f"{r['variant']} / {r['runtime']}",fmt(r['size_bytes']/2**20),
         fmt(r['benchmark']['median_ms']),fmt(r['benchmark']['p95_ms']),
         fmt(r.get('metrics_vs_torch',{}).get('snr_db','khác output contract')),
         fmt(r.get('metrics_vs_torch',{}).get('max_abs_error','—'),5),r['demo_line_count']] for r in mlsd])
    channel_table=table(['Raw variant','Center SNR dB','Displacement SNR dB','Scale output'],[
        [r['variant'],fmt(r['center_logit_metrics']['snr_db']),fmt(r['displacement_metrics']['snr_db']),
         fmt(r['output_contract'][0]['quantization'][0],6)] for r in mlsd if 'center_logit_metrics' in r])
    strict=next((r for r in distil if r['variant']=='strict'),None)
    mask100=next((r for r in distil if r['variant']=='mixed_mask1e2'),None)
    baseline=next((r for r in distil if r['variant']=='baseline'),None)
    candidates=[]
    for task,threshold in [('cv',.73),('text',.91)]:
        eligible=[r for r in runtime if r['task']==task and '_new_' in r['name'] and r['metrics']['accuracy']>=threshold]
        if eligible:
            best=min(eligible,key=lambda r:r['benchmark']['median_ms'])
            candidates.append(f"{task}: `{best['variant']}` mới, {fmt(best['benchmark']['median_ms'])} ms, accuracy {best['metrics']['accuracy']:.0%}; ứng viên để đánh giá thêm, chưa kết luận deployment.")
    strict_text=(f"Bản strict convert thành công; audit {strict['audit']['tensor_counts']}. Float còn ở CAST→QUANTIZE của mask, không phải toàn attention/LayerNorm FP32. "
                 f"Accuracy {strict['metrics']['accuracy']:.0%}, chưa chứng minh strict tốt hơn mixed."
                 if strict else 'Chưa có phép đo strict thành công; xem conversion JSON.')
    ablation_text=(f"Mask 1e30 → 1e2: accuracy {baseline['metrics']['accuracy']:.0%} → {mask100['metrics']['accuracy']:.0%}, "
                   f"SNR {fmt(baseline['metrics']['snr_db'])} → {fmt(mask100['metrics']['snr_db'])} dB."
                   if mask100 and baseline else 'Đối chứng mask chưa có đầy đủ số đo.')
    bridge_path=DAY/'experiments/03-mlsd/results/run-01/bridge_validation.json'
    bridge=read(bridge_path) if bridge_path.exists() else {}
    execution_path=DAY/'experiments/01-reproduce/results/run-01/execution.json'
    execution=read(execution_path) if execution_path.exists() else []
    failed=[r['name'] for r in execution if r['exit_code']]
    controls=[read(p) for p in sorted((DAY/'experiments/01-reproduce/results/run-02-threadcheck').glob('text_*.json'))]
    controls_text=table(['TF global thread setting','Median ms','Accuracy','SNR dB'],[
        ['set API' if r['benchmark'].get('initialize_tf_threads') else 'environment only',
         fmt(r['benchmark']['median_ms']),f"{r['metrics']['accuracy']:.0%}",fmt(r['metrics']['snr_db'])] for r in controls])
    boundaries=[]
    for runtime_name in ('old','new'):
        p=DAY/f'experiments/01-reproduce/results/run-01/delegate_audit_{runtime_name}.json'
        if p.exists():
            r=read(p);boundaries.append(f"{runtime_name}: {len(r['delegate_boundaries'])} DELEGATE nodes")
    box_path=DAY/'experiments/04-box-postprocess/results/run-01/demo_boxes.json'
    box=read(box_path) if box_path.exists() else None
    box_text=(f"Smoke test desktop từ {box['input_segments']} line chưa merge: {len(box['boxes'])} box ứng viên, median {fmt(box['median_ms'])} ms. "
              'Đây là kết quả prototype, không phải chất lượng NAVER hoặc tốc độ Android.' if box else 'Chưa có smoke test input thực tế.')
    run_times=[]
    for p in sorted((DAY/'experiments/01-reproduce/results/run-01').glob('*.log')):
        run_times.append({'log':p.name,
                          'file_created_at':datetime.datetime.fromtimestamp(p.stat().st_ctime,datetime.timezone(datetime.timedelta(hours=7))).isoformat(),
                          'file_modified_at':datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone(datetime.timedelta(hours=7))).isoformat()})
    (DAY/'experiments/01-reproduce/results/run-01/run_times.json').write_text(json.dumps(
        {'timezone':'Asia/Saigon','scope':'Windows log-file creation/modification times, approximate subprocess boundaries','jobs':run_times},indent=2),encoding='utf-8')
    content=f'''# Báo cáo thực nghiệm ngày 09/10/2026

**Người thực hiện:** Huy. **Máy:** Windows x64, i7-12700H, RAM 16 GB. **Phạm vi:** CPU; chưa chạy M-LSD hoặc post-processing trên Android. Code và tài liệu thuật toán có hỗ trợ AI, Huy cần tự đọc lại trước khi trình bày.

## 1. Mục tiêu và kết quả bàn giao

- Đọc repo Đức tại revision `536b9ae1f9e1b3829ac4776a98fb05677df9b6aa`, giữ nguyên repo; chạy **model của Huy** trên runtime cũ và runtime mới tương ứng phiên bản Đức khai báo.
- Đối chứng DistilBERT: mixed static INT8, strict INT8 và hai giá trị attention-mask sentinel `1e4`, `1e2`.
- Chuyển **trọng số PyTorch M-LSD Tiny 512** sang Keras tương đương rồi TFLite; thử FP16 weights, dynamic range, static INT8 và graph chứa decoder.
- Prototype hình học Java có 5 ca kiểm tra; đề xuất contract Android và những phần cần port tiếp.

**Giới hạn reproduce:** chưa chạy lại đầy đủ ResNet18/pipeline converter của Đức và chưa reproduce phần cứng Mac M1 Pro. Pip cài `litert-torch==0.9.4` thất bại vì dependency `litert-converter==0.4.*` không có distribution phù hợp trên Windows. Không coi đọc code hoặc cài một phần dependency là reproduce toàn bộ. Bảng bên dưới là đối chứng runtime trên cùng Windows, cùng model Huy.

## 2. Đối chiếu môi trường và giao thức

| Nội dung | Đức theo source/báo cáo | Huy trong phiên hôm nay |
| --- | --- | --- |
| Model ảnh | ResNet18 torchvision, logits | ResNet50 Keras, softmax |
| Model text | Không có trong bài đối chiếu | DistilBERT SST-2 |
| TFLite quantizer | LiteRT Torch → AI Edge Quantizer recipes | TensorFlow converter 2.15.1 |
| Runtime | ORT 1.30.0, LiteRT 2.2.0 | Cũ: ORT 1.20.1/TF 2.15.1; mới: ORT 1.30.0/LiteRT 2.2.0 |
| Phần cứng | Mac M1 Pro theo báo cáo | i7-12700H Windows |
| Calibration/evaluation | 100/300 ảnh | Model Huy giữ nguyên 100/100; M-LSD 32/10 ảnh |
| Thread/warm-up/runs | 4 / 10 / 100 | 4 / 30 / 200; ORT inter-op=1 |

Mỗi biến thể chạy process riêng, tuần tự; batch 1. Timer gồm API đồng bộ, copy tensor và quantize/dequantize I/O; loại load model, file và preprocessing. Benchmark lặp input đầu tiên; chưa đo phân phối nhiều loại input, nhiệt độ hay công suất. JSON lưu 200 mẫu latency, hash model/input/output. TFLite dùng delegate mặc định của Interpreter; log ghi XNNPACK nếu được tạo, chưa cố định mọi partition của delegate.

Nguồn Đức: [prepare](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q1_prepare.py), [quantizer](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q3_quant_tflite.py), [benchmark](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/q4_evaluate.py). Claim range/ties trong [báo cáo Đức](https://github.com/TruongDuke/Quantization/blob/536b9ae1f9e1b3829ac4776a98fb05677df9b6aa/4-quantization/BAO_CAO.md) chưa được xác minh từ raw output vì repo không chứa artifact đó.

## 3. Cùng model Huy, thay runtime

{runtime_table}

Tỷ lệ mới/cũ <1 nghĩa là nhanh hơn. Đây là thay **bộ software/runtime**: hai môi trường còn khác NumPy và dependency native, vì vậy chưa quy toàn bộ thay đổi cho một kernel hay một phiên bản ORT. Không đem trực tiếp latency ResNet18 Mac so với ResNet50 Windows để kết luận framework nhanh hơn. Cùng 4 thread cũng khác kiến trúc ARM/x86, SIMD, bộ nhớ, scheduling, fusion và khả năng delegate xử lý operator.

Một khác biệt phần mềm có cơ sở: TensorFlow công bố XNNPACK hỗ trợ dynamic-range Fully Connected/Conv2D và bật mặc định trong prebuilt binaries từ TF 2.17; baseline TF 2.15 nằm trước thay đổi này. Đây là **giả thuyết góp phần** giải thích dynamic chênh lệch, chưa phải kết quả profiling xác định từng kernel trong hai wheel đang dùng. [Thông báo TensorFlow](https://blog.tensorflow.org/2024/04/faster-dynamically-quantized-inference-with-xnnpack.html). FP32 chậm hơn ở runtime mới vẫn cần profile/thread-power đối chứng riêng.

Audit ResNet50 dynamic sau phép đo: {'; '.join(boundaries)}. [Audit cũ](experiments/01-reproduce/results/run-01/delegate_audit_old.json), [audit mới](experiments/01-reproduce/results/run-01/delegate_audit_new.json) lưu ranh giới tensor. Introspection dùng API private, có cả original nodes và delegate nodes, chưa xác nhận coverage hoặc timing từng kernel.

**Đối chứng do latency biến động:** DistilBERT static cùng file có số đo khác giữa nhóm ablation và runtime matrix. Chạy tiếp hai process liền nhau với cùng script/model/input, chỉ đổi cách đặt TF global threads; Interpreter vẫn 4 thread trong cả hai:

{controls_text}

Source/JSON: [run-02-threadcheck](experiments/01-reproduce/results/run-02-threadcheck/). Matrix đặt TF thread qua environment; ablation đặt thêm API. Khác biệt protocol này và trạng thái máy là giới hạn khi so hai nhóm; không coi mọi thay đổi latency là do phiên bản thư viện. Chưa profile nhiệt độ/power hoặc lặp nhiều session.

Cặp kiểm tra liền nhau không tái hiện lợi thế latency 43 ms của nhóm ablation; đổi sang API đặt thread không khắc phục chênh lệch. Vì vậy chưa xác định nguyên nhân biến động giữa các phase. Các bảng latency là snapshot của phiên đo, cần kiểm soát power/nhiệt độ và lặp lại để đưa ra kết luận hiệu năng chắc hơn; không so chéo phase như cùng trạng thái máy.

Accuracy được tính trên 100 mẫu có nhãn thật. Sai số output không đồng nghĩa với đổi nhãn; JSON từng biến thể còn lưu agreement, MAE, max error, relative L2, cosine và SNR so với TensorFlow gốc. Các tỷ lệ này chỉ mô tả tập nhỏ đã chọn.

Ứng viên nhanh nhất trong các row runtime mới có accuracy không thấp hơn reference trên tập này: {' '.join(candidates)} Không dùng chênh lệch 1–2 mẫu để kết luận accuracy cải thiện có ý nghĩa thống kê.

## 4. Dynamic/static ONNX Runtime và TFLite/LiteRT

`x ≈ scale × (q − zero_point)`. Dynamic tính range activation được hỗ trợ trong lúc inference; static dùng range từ representative/calibration data rồi cố định khi chạy. Cả hai đều có thể giữ operator float. [ONNX Runtime](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).

TFLite dynamic range thường lưu weights INT8, dùng hybrid kernels ở operator hỗ trợ; activation hoặc phần graph khác có thể vẫn float. Static calibration có thể cho graph mixed hoặc ép integer-only, tùy cấu hình converter. Token IDs/index/bias INT32 không phải float fallback. [LiteRT PTQ](https://developers.google.com/edge/litert/conversion/tensorflow/quantization/post_training_quantization).

Cùng tên phương pháp vẫn khác: operator được chọn, MinMax/histogram, per-tensor/per-channel, signedness, fusion, biểu diễn QDQ/QOperator và kernel/provider/delegate. Bài Huy ONNX text chọn MatMul/Gemm; TFLite có phạm vi khác. Đức quantize file TFLite bằng AI Edge Quantizer, không phải cùng TensorFlow converter của Huy. Phải audit graph thay vì suy từ nhãn “INT8”. Calibration không đại diện evaluation có thể ảnh hưởng **cả ONNX và TFLite**; chưa có chứng cứ ONNX miễn nhiễm.

Ở ONNX QDQ text của Huy, mask/Sub/Softmax nằm ngoài danh sách operator quantize; audit vẫn có 6 Softmax và nhiều operator float. TFLite static phủ rộng hơn, kể cả vùng mask với scale rất lớn. Vì vậy cùng “static” chưa phải hai graph có cùng rủi ro số học. QDQ giữ tên MatMul float trong graph lưu trữ, runtime có thể fuse kernel INT8; chỉ đếm tên node chưa đủ để kết luận dtype tính toán. [Metadata ONNX](../../quantization/results/text/conversion_onnx_static.json).

## 5. Phần riêng Huy: DistilBERT calibrated mixed INT8

**Calibrated mixed INT8 là static quantization cho phép graph mixed**, không phải phương pháp thứ ba nằm giữa dynamic và static. Static/dynamic nói lúc lấy scale; mixed/full nói phạm vi dtype/operator.

Cấu hình ngày trước dùng representative dataset, cho phép `TFLITE_BUILTINS_INT8` + `TFLITE_BUILTINS`, output INT8, IDs/mask INT32. Tensor float còn lại xác định là `distilbert/Cast [1,64]`; không có bằng chứng đã chủ động giữ LayerNorm hoặc toàn attention FP32.

{strict_text} Vì vậy lý do chọn mixed trước đây là cho phép fallback để converter dễ chạy, **chưa phải lý do thực nghiệm chứng minh mixed cần thiết hoặc giữ chất lượng tốt hơn**. Cho phép mixed cũng không tự bảo vệ phần nhạy cảm khỏi bị quantize.

{distil_table}

Hướng thử mới: chỉ thay sentinel attention mask `1e30` bằng `1e4` hoặc `1e2`, giữ weights/input/calibration. Trên 100 evaluation, hai bản FP32 đã so với reference và cho output giống hệt (max error=0, accuracy 91%). Đây là kiểm tra trên tập đã dùng, chưa bảo đảm cho mọi sequence/input.

{ablation_text} Bản mask 1e2 vẫn thấp hơn FP32 91%, chưa đạt chất lượng để khuyến nghị deployment. Audit scale đi cùng output giúp đánh giá giả thuyết sentinel quá lớn làm resolution INT8 quanh attention quá thô. Các giá trị ablation được chọn trước phép đo; chưa tune trên tập test độc lập. Không gọi mọi chênh lệch là lỗi calibration của thư viện, hoặc kết luận đây là nguyên nhân duy nhất.

Evidence/code: [run.py](experiments/02-distilbert-int8/src/run.py), [kết quả](experiments/02-distilbert-int8/results/run-01/).

## 6. M-LSD Tiny: chuyển đổi và quantization

PyTorch revision `2312205254e66911703decf775f626995d260f17`; NAVER revision `453cafa09467d0272760578d35c1fda38e8895a5`. [Source/hash lock](experiments/03-mlsd/sources.lock.json). Không huấn luyện lại. Copy OIHW→HWIO, depthwise, BN eps, pad phải/dưới cho stride 2 và bilinear `align_corners=True`.

Input raw RGBA 512×512, alpha=1; model bridge normalize `x/127.5−1`. PyTorch nhận NCHW đã normalize. Reference raw là NHWC `[N,256,256,9]`. Bản TFLite chính thức trả `points [1,200,2]`, `scores [1,200]`, `vmap [1,256,256,4]`.

**Kiểm tra bridge:** SNR {fmt(bridge.get('snr_db'))} dB, relative L2 {fmt(bridge.get('relative_l2'),8)}, max raw error {fmt(bridge.get('max_abs_error'),6)}. `allclose(atol=rtol=1e-4)` ban đầu không đạt. Sau đọc mapping/architecture và đối chứng kernel, dùng gate công khai relative L2<1e-4 và max error<0,1 map unit; với displacement tương ứng <0,2 pixel ảnh 512. Đây là ngưỡng chất lượng số, không phải bit-equivalence và chưa chứng minh accuracy detect box. File validation giữ cả kết quả ngưỡng chặt và ngưỡng hình học.

{mlsd_table}

{channel_table}

Scale 0 ở output float nghĩa là không có quantization I/O. Raw head có center logit và displacement khác miền giá trị trong cùng tensor; INT8 per-tensor output dùng chung scale, có thể làm center mất resolution. Đây là giả thuyết cần đối chứng tách head/quantization chọn lọc, không phải nguyên nhân duy nhất đã xác định. FP16 weights làm nhỏ file nhưng số liệu hiện tại cho thấy sai số tăng; chưa có cơ sở bảo đảm chất lượng detection tương đương FP32.

Chỉ đối chiếu tốc độ **decoded_fp32 ↔ official** cùng runtime/thread vì hai graph cùng trả decoder outputs. Raw FP32/FP16/dynamic/static có scope khác và chỉ so trong nhóm raw. Torch reference 1 thread, input đã normalize, nên latency Torch riêng không phải so công bằng với bảng 4 thread TFLite.

Sai số conversion/quantization luôn so với **cùng checkpoint PyTorch**, không dùng official khác checkpoint/graph làm chuẩn. Calibration: 32 ảnh train Imagenette, evaluation: 10 ảnh validation, không trùng hash, seed 20261009. Đây là ảnh thật nhưng chưa phải dataset có nhãn line/box; chưa có precision/recall, sAP hoặc IoU box. SNR toàn raw output có thể che sai số center/displacement; JSON lưu thêm metrics hai nhóm riêng. Line count chỉ là kiểm tra pipeline, không phải accuracy.

Evidence/code: [prepare_torch.py](experiments/03-mlsd/src/prepare_torch.py), [port_keras.py](experiments/03-mlsd/src/port_keras.py), [measure.py](experiments/03-mlsd/src/measure.py), [kết quả](experiments/03-mlsd/results/run-01/).

## 7. Detect box và hướng Android

Luồng source NAVER: decode line → Hough merge → giao điểm → kiểm tra khoảng cách/góc → ghép chu trình bốn cạnh → chấm điểm. Center NMS 3×3; Hough accumulator suppression 5×5. Tọa độ point là (y,x), displacement là (x,y).

[Tài liệu đọc thuật toán và contract Android](experiments/04-box-postprocess/ALGORITHM.md). [Prototype Java](experiments/04-box-postprocess/src/BoxPostProcessor.java) xử lý giao điểm/chu trình, giới hạn 64 line, qua 5 ca hình học. Chưa port Hough merge và scoring tương đương NAVER. Thử CSV demo chỉ kiểm tra nối pipeline vì CSV chưa merge; latency JVM Windows không phải latency Android.

{box_text} [Evidence Java](experiments/04-box-postprocess/results/run-01/demo_boxes.json).

Đề xuất triển khai Kotlin/Java với primitive arrays và buffer tái sử dụng trước; chỉ chuyển C++/JNI khi profiler máy thật chứng minh bottleneck. Cần fixtures đối chiếu góc/box, kiểm tra song song/đoạn bằng 0/NaN và biến đổi ngược crop/letterbox.

## 8. Hướng tiếp và phần còn thiếu

1. Đối chứng sentinel đã chạy là hướng thử nghiệm bổ sung hôm nay. Tiếp theo dùng test set độc lập/sequence length khác để kiểm tra khả năng tổng quát; thử quantize chọn lọc attention hoặc QAT nếu PTQ vẫn giảm chất lượng.
2. Để hoàn tất reproduce Đức: cần Linux tương thích converter hoặc artifact đã export cùng version/hash từ Đức. Đối chiếu ResNet18 cùng data/preprocess trước khi so với model khác. Chưa có phép đo Mac M1 Pro độc lập.
3. M-LSD: dùng calibration cùng miền ảnh đường thẳng và dataset có nhãn; đánh giá sAP/IoU sau decoding, thử nhiều subset mà không chọn theo test score.
4. Port Hough merge/scoring, đối chiếu fixtures với source, rồi APK benchmark trên Android thật; đo model/decoder/box/end-to-end riêng.
5. Huy tự đọc và viết lại phần thuật toán theo ý hiểu để đáp ứng yêu cầu mentor; tài liệu hỗ trợ không thay việc này.

## 9. Tái lập và kiểm chứng

[Hướng dẫn chạy](RUNNING.md), [dependency/environment](ENVIRONMENT.md), [matrix](run_matrix.py). Runner chạy tuần tự và ghi exit code từng job. Số job đã có trạng thái: {len(execution)}; thất bại: {len(failed)} ({', '.join(failed) if failed else 'không có trong các job đã ghi'}). Không coi “chưa đo” là 0 ms.

Metric dùng float64: `SNR=10·log10(sum(ref²)/sum((ref−test)²))`; MAE/max đo trên output dequantized, cosine và relative L2 toàn tensor. Median/p95 tính lại từ raw latency samples. ZIP chứa code, báo cáo và bằng chứng; model/venv/cache/credentials không đưa vào Git. Bản chat cục bộ được ignore theo cấu hình repo.
'''
    (DAY/'REPORT.md').write_text(content,encoding='utf-8')
    paired={}
    for version in ('2.15.1','2.2.0'):
        own=next((r for r in mlsd if r['variant']=='decoded_fp32' and r['runtime']==version),None)
        official=next((r for r in mlsd if r['variant']=='official' and r['runtime']==version),None)
        if own and official:paired[version]=f"{own['benchmark']['median_ms']:.1f} vs {official['benchmark']['median_ms']:.1f} ms"
    static_mlsd=next((r for r in mlsd if r['variant']=='static'),None)
    static_note=(f"Static INT8 còn {static_mlsd['size_bytes']/2**20:.2f} MiB nhưng SNR {static_mlsd['metrics_vs_torch']['snr_db']:.2f} dB, chưa đạt chất lượng."
                 if static_mlsd else 'Chưa có phép đo static M-LSD.')
    chat=f'''Anh ơi em báo cáo ngày 09/10 ạ.
- Em đọc repo Đức, đối chiếu pipeline; đã chạy ResNet50/DistilBERT của em với ORT 1.20.1→1.30.0 và TF Lite 2.15.1→LiteRT 2.2.0 trên cùng máy Windows, cùng input và 4 thread. Chưa reproduce đầy đủ converter/ResNet18 Mac vì thiếu dependency converter phù hợp trên Windows.
- Calibrated mixed INT8 là static quantization cho phép float fallback. {strict_text}
- Em thử đổi attention mask sentinel, kiểm tra FP32 vẫn giống reference trên 100 mẫu. {ablation_text}
- M-LSD Tiny cùng checkpoint PyTorch→TFLite, đã thử FP16/dynamic/static. So graph có decoder với official: TF Lite {paired.get('2.15.1','chưa đo')}; LiteRT {paired.get('2.2.0','chưa đo')} (CPU 4 thread, 30 warm-up/200 lượt). Latency là snapshot, chưa kiểm soát nhiệt độ/power. {static_note} Chưa có metric line/box trên dataset có nhãn.
- Prototype Java qua 5 ca hình học, demo được 4 box ứng viên; chưa port đầy đủ Hough merge/scoring hoặc benchmark Android. Em cần tự đọc lại thuật toán trước khi trình bày.
- Bàn giao code, JSON/CSV kết quả, báo cáo MD và ZIP. Hướng tiếp: calibration đúng miền M-LSD, dataset có nhãn và test sentinel trên tập độc lập.
'''
    (DAY/'REPORT-CHAT.md').write_text(chat,encoding='utf-8')
    with (DAY/'experiments/01-reproduce/results/run-01/runtime_summary.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['name','runtime','accuracy','snr_db','median_ms','p95_ms'])
        for r in runtime:writer.writerow([r['name'],r['runtime']['version'],r['metrics']['accuracy'],r['metrics']['snr_db'],r['benchmark']['median_ms'],r['benchmark']['p95_ms']])
    print(f'Report built: {len(runtime)} runtime rows, {len(distil)} DistilBERT rows, {len(mlsd)} M-LSD rows')


if __name__=='__main__':main()
