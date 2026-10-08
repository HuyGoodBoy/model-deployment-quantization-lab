package com.vin.deploylab;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.Build;
import android.os.Bundle;
import android.os.PowerManager;
import android.os.SystemClock;
import android.util.Log;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;

import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public final class BenchmarkTest {
    private static final int INPUT_FLOATS = 224 * 224 * 3;
    private static final int OUTPUT_FLOATS = 1000;
    private Context context;
    private File directory;

    private interface Runner extends AutoCloseable {
        float[] run(byte[] input) throws Exception;
        JSONObject info() throws Exception;
    }

    @Test public void benchmarkCpu() throws Exception {
        context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        directory = new File(context.getExternalFilesDir(null), "benchmark");
        assertTrue("Cannot create output directory", directory.isDirectory() || directory.mkdirs());
        Bundle arguments = InstrumentationRegistry.getArguments();
        int threads = positive(arguments, "threads", 1);
        int warmup = positive(arguments, "warmup", 30);
        int runs = positive(arguments, "runs", 200);
        assertTrue("Too many threads", threads <= 8);
        JSONObject bundle = new JSONObject(new String(asset("bundle.json"), StandardCharsets.UTF_8));
        assertEquals("float32", bundle.getString("dtype"));
        assertEquals(ByteOrder.LITTLE_ENDIAN, ByteOrder.nativeOrder());
        JSONArray records = bundle.getJSONArray("records");
        assertTrue("No inputs", records.length() > 0);
        JSONObject report = new JSONObject();
        report.put("schema_version", 1);
        report.put("started_at_unix_ms", System.currentTimeMillis());
        report.put("bundle", bundle);
        report.put("system", system());
        report.put("threads", threads);
        report.put("warmup", warmup);
        report.put("runs", runs);
        report.put("batch_size", 1);
        report.put("timing_scope", "Java inference API including input/output copies and tensor creation; excludes model/file load, preprocessing, file write and metric calculation");
        String[] order = "tflite-first".equals(arguments.getString("runtime_order"))
            ? new String[]{"tflite", "onnx"} : new String[]{"onnx", "tflite"};
        report.put("runtime_order", new JSONArray(Arrays.asList(order)));
        JSONObject benchmarks = new JSONObject();
        report.put("benchmarks", benchmarks);
        for (String runtime : order) {
            JSONObject model = bundle.getJSONObject("models").getJSONObject(runtime);
            byte[] modelBytes = asset(model.getString("asset"));
            assertEquals("Model hash mismatch", model.getString("sha256"), sha256(modelBytes));
            Runner runner = runtime.equals("onnx") ? new OnnxRunner(modelBytes, threads) : new LiteRunner(modelBytes, threads);
            try {
                JSONObject stats = runner.info();
                JSONArray outputRecords = new JSONArray();
                for (int i = 0; i < records.length(); i++) {
                    JSONObject record = records.getJSONObject(i);
                    byte[] input = input(record);
                    float[] output = runner.run(input);
                    validateOutput(output);
                    String outputFile = runtime + "_" + i + ".f32";
                    byte[] outputBytes = floatBytes(output);
                    save(outputFile, outputBytes);
                    outputRecords.put(new JSONObject().put("filename", record.getString("filename"))
                        .put("file", outputFile).put("sha256", sha256(outputBytes)));
                }
                stats.put("outputs", outputRecords);
                byte[] input = input(records.getJSONObject(0));
                stats.put("input_image", records.getJSONObject(0).getString("filename"));
                stats.put("before", system());
                for (int i = 0; i < warmup; i++) runner.run(input);
                double[] samples = new double[runs];
                float[] output = null;
                for (int i = 0; i < runs; i++) {
                    long start = SystemClock.elapsedRealtimeNanos();
                    output = runner.run(input);
                    long stop = SystemClock.elapsedRealtimeNanos();
                    samples[i] = (stop - start) / 1e6;
                }
                validateOutput(output);
                stats.put("after", system());
                double mean = 0;
                for (double sample : samples) mean += sample;
                mean /= samples.length;
                double[] sorted = samples.clone();
                Arrays.sort(sorted);
                stats.put("mean_ms", mean).put("median_ms", percentile(sorted, 50))
                    .put("p95_ms", percentile(sorted, 95)).put("images_per_second", 1000.0 / mean);
                JSONArray allSamples = new JSONArray();
                for (double sample : samples) allSamples.put(sample);
                stats.put("latency_samples_ms", allSamples);
                benchmarks.put(runtime, stats);
                // Persist after each runtime so failures don't erase already collected evidence.
                save("android_run.json", report.toString(2).getBytes(StandardCharsets.UTF_8));
                Log.i("DeployLab", runtime + " mean_ms=" + mean + " median_ms=" + percentile(sorted, 50));
            } finally {
                runner.close();
            }
        }
        assertEquals(2, benchmarks.length());
        Log.i("DeployLab", "RESULTS_DIR=" + directory.getAbsolutePath());
    }

    private static int positive(Bundle args, String key, int fallback) {
        int value = Integer.parseInt(args.getString(key, String.valueOf(fallback)));
        if (value < 1 || value > 10000) throw new IllegalArgumentException("Invalid " + key);
        return value;
    }

    private byte[] asset(String name) throws Exception {
        try (InputStream stream = context.getAssets().open("lab/" + name);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[65536];
            int size;
            while ((size = stream.read(buffer)) != -1) output.write(buffer, 0, size);
            return output.toByteArray();
        }
    }

    private byte[] input(JSONObject record) throws Exception {
        byte[] bytes = asset(record.getString("input_asset"));
        assertEquals(INPUT_FLOATS * 4, bytes.length);
        assertEquals("Input hash mismatch", record.getString("input_sha256"), sha256(bytes));
        return bytes;
    }

    private void save(String name, byte[] bytes) throws Exception {
        try (FileOutputStream output = new FileOutputStream(new File(directory, name))) {
            output.write(bytes);
        }
    }

    private static String sha256(byte[] bytes) throws Exception {
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder result = new StringBuilder();
        for (byte b : digest) result.append(String.format("%02x", b & 255));
        return result.toString();
    }

    private static byte[] floatBytes(float[] values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        return bytes.array();
    }

    private static void validateOutput(float[] output) {
        assertNotNull(output);
        assertEquals(OUTPUT_FLOATS, output.length);
        double sum = 0;
        for (float value : output) {
            assertTrue("Non-finite output", Float.isFinite(value));
            assertTrue("Not probabilities", value >= -1e-6 && value <= 1.000001);
            sum += value;
        }
        assertEquals("Softmax output does not sum to 1", 1, sum, 1e-4);
    }

    private JSONObject system() throws Exception {
        JSONObject info = new JSONObject().put("manufacturer", Build.MANUFACTURER)
            .put("model", Build.MODEL).put("device", Build.DEVICE).put("hardware", Build.HARDWARE)
            .put("android_release", Build.VERSION.RELEASE).put("android_api", Build.VERSION.SDK_INT)
            .put("abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS))).put("fingerprint", Build.FINGERPRINT);
        Intent battery = context.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
        if (battery != null) info.put("battery_temperature_c", battery.getIntExtra("temperature", -1) / 10.0);
        if (Build.VERSION.SDK_INT >= 29) info.put("thermal_status", ((PowerManager) context.getSystemService(Context.POWER_SERVICE)).getCurrentThermalStatus());
        return info;
    }

    private static double percentile(double[] sorted, double percentage) {
        double position = (sorted.length - 1) * percentage / 100;
        int lower = (int) Math.floor(position), upper = (int) Math.ceil(position);
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
    }

    private static final class LiteRunner implements Runner {
        private final Interpreter interpreter;
        private final ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_FLOATS * 4).order(ByteOrder.nativeOrder());
        private final ByteBuffer outputBuffer = ByteBuffer.allocateDirect(OUTPUT_FLOATS * 4).order(ByteOrder.nativeOrder());
        private final float[] output = new float[OUTPUT_FLOATS];
        private final ByteBuffer modelBuffer;

        LiteRunner(byte[] model, int threads) {
            modelBuffer = ByteBuffer.allocateDirect(model.length).order(ByteOrder.nativeOrder());
            modelBuffer.put(model).rewind();
            interpreter = new Interpreter(modelBuffer, new Interpreter.Options()
                .setNumThreads(threads).setUseXNNPACK(true).setUseNNAPI(false));
            interpreter.allocateTensors();
            assertArrayEquals(new int[]{1,224,224,3}, interpreter.getInputTensor(0).shape());
            assertArrayEquals(new int[]{1,1000}, interpreter.getOutputTensor(0).shape());
            assertEquals(DataType.FLOAT32, interpreter.getInputTensor(0).dataType());
            assertEquals(DataType.FLOAT32, interpreter.getOutputTensor(0).dataType());
        }

        public float[] run(byte[] input) {
            inputBuffer.clear();
            inputBuffer.put(input).rewind();
            outputBuffer.clear();
            interpreter.run(inputBuffer, outputBuffer);
            outputBuffer.rewind();
            outputBuffer.asFloatBuffer().get(output);
            return output;
        }

        public JSONObject info() throws Exception {
            return new JSONObject().put("version", "2.15.0").put("mode", "TensorFlow Lite Java Interpreter")
                .put("delegate", "XNNPACK requested; unsupported operators may use builtin CPU kernels")
                .put("use_xnnpack", true).put("use_nnapi", false).put("use_gpu", false);
        }
        public void close() { interpreter.close(); }
    }

    private static final class OnnxRunner implements Runner {
        private final OrtEnvironment environment = OrtEnvironment.getEnvironment();
        private final OrtSession.SessionOptions options = new OrtSession.SessionOptions();
        private final OrtSession session;
        private final String inputName;
        private final ByteBuffer inputBuffer = ByteBuffer.allocateDirect(INPUT_FLOATS * 4).order(ByteOrder.nativeOrder());
        private final float[] output = new float[OUTPUT_FLOATS];

        OnnxRunner(byte[] model, int threads) throws Exception {
            options.setIntraOpNumThreads(threads);
            options.setInterOpNumThreads(1);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            session = environment.createSession(model, options);
            assertEquals(1, session.getInputNames().size());
            inputName = session.getInputNames().iterator().next();
            TensorInfo inputInfo = (TensorInfo) session.getInputInfo().get(inputName).getInfo();
            assertArrayEquals(new long[]{1,224,224,3}, inputInfo.getShape());
            assertEquals(OnnxJavaType.FLOAT, inputInfo.type);
            TensorInfo outputInfo = (TensorInfo) session.getOutputInfo().values().iterator().next().getInfo();
            assertArrayEquals(new long[]{1,1000}, outputInfo.getShape());
            assertEquals(OnnxJavaType.FLOAT, outputInfo.type);
        }

        public float[] run(byte[] input) throws Exception {
            inputBuffer.clear();
            inputBuffer.put(input).rewind();
            try (OnnxTensor tensor = OnnxTensor.createTensor(environment, inputBuffer.asFloatBuffer(), new long[]{1,224,224,3});
                 OrtSession.Result result = session.run(Collections.singletonMap(inputName, tensor))) {
                float[][] values = (float[][]) result.get(0).getValue();
                System.arraycopy(values[0], 0, output, 0, OUTPUT_FLOATS);
            }
            return output;
        }

        public JSONObject info() throws Exception {
            return new JSONObject().put("version", "1.20.0").put("mode", "ONNX Runtime Android Java")
                .put("provider", "CPUExecutionProvider").put("graph_optimization", "ORT_ENABLE_ALL")
                .put("use_nnapi", false).put("use_gpu", false);
        }
        public void close() throws Exception { session.close(); options.close(); }
    }
}
