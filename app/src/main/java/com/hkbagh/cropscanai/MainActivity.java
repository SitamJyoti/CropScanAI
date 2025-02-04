package com.hkbagh.cropscanai;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp;
import org.tensorflow.lite.support.label.TensorLabel;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int CAMERA_PERMISSION_REQUEST_CODE = 100;
    private static final int CAMERA_REQUEST_CODE = 101;
    private static final int GALLERY_REQUEST_CODE = 102;
    private ImageView imageView;
    private Button captureButton;
    private TextView resultTextView;
    private Interpreter tflite;
    private List<String> labels;
    private Bitmap inputBitmap;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        imageView = findViewById(R.id.imageView);
        captureButton = findViewById(R.id.captureButton);
        resultTextView = findViewById(R.id.resultTextView);

        try {
            tflite = new Interpreter(FileUtil.loadMappedFile(this, "your_model.tflite")); // Replace with your model file name
            labels = FileUtil.loadLabels(this, "labels.txt"); // Replace with your labels file name
        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(this, "Error loading model or labels", Toast.LENGTH_SHORT).show(); // Handle the error
        }

        captureButton.setOnClickListener(v -> showImagePickerDialog());
    }
    private void showImagePickerDialog() {
        final CharSequence[] options = {"Take Photo", "Choose from Gallery", "Cancel"};
        androidx.appcompat.app.AlertDialog.Builder builder = new androidx.appcompat.app.AlertDialog.Builder(this);
        builder.setTitle("Add Photo!");
        builder.setItems(options, (dialog, item) -> {
            if (options[item].equals("Take Photo")) {
                checkCameraPermissionAndOpenCamera();
            } else if (options[item].equals("Choose from Gallery")) {
                openGallery();
            } else if (options[item].equals("Cancel")) {
                dialog.dismiss();
            }
        });
        builder.show();
    }


    private void checkCameraPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            requestPermissions(new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST_CODE);
        }
    }

    private void openCamera() {
        Intent intent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(intent, CAMERA_REQUEST_CODE);
    }

    private void openGallery() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), GALLERY_REQUEST_CODE);
    }




    @Override
    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        if (requestCode == CAMERA_REQUEST_CODE && resultCode == RESULT_OK) {
            Bitmap photo = (Bitmap) data.getExtras().get("data");
            inputBitmap = photo;
            imageView.setImageBitmap(photo);
            classifyImage(photo);

        } else if (requestCode == GALLERY_REQUEST_CODE && resultCode == RESULT_OK && data != null) {
            Uri selectedImage = data.getData();
            try {
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImage);
                inputBitmap = bitmap;
                imageView.setImageBitmap(bitmap);
                classifyImage(bitmap);
            } catch (IOException e) {
                e.printStackTrace();
                Toast.makeText(this, "Error loading image", Toast.LENGTH_SHORT).show(); // Handle the error
            }
        }
    }

    private void classifyImage(Bitmap bitmap) {
        if (tflite == null || labels == null) {
            Log.e("TFLite", "TFLite model or labels not loaded");
            return;
        }

        int imageSizeX = tflite.getInputTensor(0).shape()[1];
        int imageSizeY = tflite.getInputTensor(0).shape()[2];

        ImageProcessor imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeWithCropOrPadOp(imageSizeX, imageSizeY))
                        .add(new ResizeOp(imageSizeX, imageSizeY, ResizeOp.ResizeMethod.BILINEAR))
                        .build();

        TensorImage inputImage = TensorImage.fromBitmap(bitmap);
        inputImage = imageProcessor.process(inputImage);


        // 1. Get output tensor information
        org.tensorflow.lite.Tensor outputTensor = tflite.getOutputTensor(0); // Use the full path
        int[] outputShape = outputTensor.shape();
        DataType outputDataType = outputTensor.dataType();

        // 2. Allocate a ByteBuffer for the output
        int outputSize = 1;
        for (int dim : outputShape) {
            outputSize *= dim;
        }

        ByteBuffer outputBuffer;
        if (outputDataType == DataType.FLOAT32) {
            outputBuffer = ByteBuffer.allocateDirect(outputSize * 4); // 4 bytes per float
        } else if (outputDataType == DataType.UINT8) {
            outputBuffer = ByteBuffer.allocateDirect(outputSize); // 1 byte per uint8
        } else {
            throw new IllegalArgumentException("Output data type " + outputDataType + " not supported yet.");
        }
        outputBuffer.order(ByteOrder.nativeOrder());

        // 3. Run inference
        tflite.run(inputImage.getBuffer(), outputBuffer);

        // 4. Access the output data from outputBuffer
        String predictedLabel = "Unknown"; // Default label
        try {
            if (outputDataType == DataType.FLOAT32) {
                FloatBuffer floatBuffer = outputBuffer.asFloatBuffer();
                float[] outputArray = new float[outputSize];
                floatBuffer.get(outputArray);

                int maxIndex = 0;
                float maxProbability = outputArray[0];
                for (int i = 1; i < outputArray.length; i++) {
                    if (outputArray[i] > maxProbability) {
                        maxIndex = i;
                        maxProbability = outputArray[i];
                    }
                }
                predictedLabel = labels.get(maxIndex);
            } else if (outputDataType == DataType.UINT8) {
                byte[] outputArray = new byte[outputSize];
                outputBuffer.get(outputArray);
                // Process UINT8 output (you'll need to adapt this based on your model)
                // Example (assuming labels correspond to indices):
                int maxIndex = 0;
                byte maxValue = outputArray[0];
                for (int i = 1; i < outputArray.length; i++) {
                    if (outputArray[i] > maxValue) {
                        maxIndex = i;
                        maxValue = outputArray[i];
                    }
                }
                predictedLabel = labels.get(maxIndex);
            }
        } catch (IndexOutOfBoundsException e) {
            Log.e("TFLite", "Error processing output: " + e.getMessage());
            Toast.makeText(this, "Error processing output", Toast.LENGTH_SHORT).show();
            return; // Exit early if there is an issue.
        }

        resultTextView.setText(predictedLabel);
    }
}