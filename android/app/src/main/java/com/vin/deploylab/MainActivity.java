package com.vin.deploylab;

import android.app.Activity;
import android.os.Bundle;
import android.widget.TextView;

public final class MainActivity extends Activity {
    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        TextView text = new TextView(this);
        text.setPadding(32, 48, 32, 32);
        text.setTextSize(18);
        text.setText("Deploy Lab\n\nMobileNetV2 · FP32\nONNX Runtime + TensorFlow Lite\n\n"
            + "Run the instrumentation test APK to measure output and latency.\n"
            + "No Firebase SDK or network access is required inside this app.");
        setContentView(text);
    }
}
