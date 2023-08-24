package com.example.diabetic_retinopathy;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.location.Location;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    private final int CAMERA_REQ_CODE = 100;
    private final int GALLERY_REQ_CODE = 200; // Request code for gallery image selection

    TextView output_disp;
    Button btn_upload,btn_camera,prediction;
    ImageView imgCamera;
    String url="https://f58a-117-250-64-132.ngrok-free.app/predict";
    Intent imgdata;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        btn_camera=findViewById(R.id.btnCamera);
        btn_upload=findViewById(R.id.btnUpload);
        prediction=findViewById(R.id.prediction);
        output_disp=findViewById(R.id.txtView);
        imgCamera=findViewById(R.id.imgview);






  //      intent for image passing

        btn_camera.setOnClickListener(view -> {
            Intent iCamera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
            startActivityForResult(iCamera, CAMERA_REQ_CODE);
        });

        // Open the gallery to select an image on button click
        btn_upload.setOnClickListener(view -> {
            Intent iGallery = new Intent(Intent.ACTION_PICK);
            iGallery.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(iGallery, GALLERY_REQ_CODE);
        });

        //Adding click Listener on Predict so that on clicking it hit API request
        prediction.setOnClickListener(view -> {
            if (imgdata != null) {
                Bitmap img = (Bitmap) (imgdata.getExtras().get("data"));

                if(imgdata==null)
            Toast.makeText(MainActivity.this,"Invalid Data",Toast.LENGTH_SHORT).show();
        else
            Toast.makeText(MainActivity.this,"Data succesfully sent to Server",Toast.LENGTH_SHORT).show();
                // Send the image to the server
                sendImageToServer(img);
            }
        });
    }




    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
//        int myInt=resultCode;
//        String myString = String.valueOf(myInt);
//        Toast.makeText(MainActivity.this,myString,Toast.LENGTH_SHORT).show();
//        if(data==null)
//            Toast.makeText(MainActivity.this,"Invalid data",Toast.LENGTH_SHORT).show();
//        else
//            Toast.makeText(MainActivity.this,"Valid",Toast.LENGTH_SHORT).show();

        if (resultCode == RESULT_OK) {
            if (requestCode == CAMERA_REQ_CODE) {
                assert data != null: "The image data is NULL";

                Bitmap img = (Bitmap) (data.getExtras().get("data"));
                imgCamera.setImageBitmap(img);
                imgdata=data;
            }
        }

        if (requestCode == GALLERY_REQ_CODE ) {
//            Toast.makeText(MainActivity.this,"gALLery",Toast.LENGTH_SHORT).show();
            assert data != null;

//              imgCamera.setImageURI(data.getData());
//            Bitmap img = (Bitmap) (data.getExtras().get("data"));
//            imgCamera.setImageBitmap(img);
//            imgdata=data;

            // Get the selected image URI from the data


//
            Uri selectedImageUri = data.getData();
//
//
//                // Convert the selected image URI to a Bitmap
//
//            Bitmap selectedImageBitmap = null;
//            try {
//                selectedImageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
//            } catch (IOException e) {
//                Toast.makeText(MainActivity.this,"URI",Toast.LENGTH_SHORT).show();
//                e.printStackTrace();
//            }
//            Toast.makeText(MainActivity.this,"before image view",Toast.LENGTH_SHORT).show();
//                // Display the selected image
////            if(selectedImageBitmap!=NULL)
//                imgCamera.setImageBitmap(selectedImageBitmap);
//                Toast.makeText(MainActivity.this,"after image view",Toast.LENGTH_SHORT).show();
//                imgdata = data; // Save the selected image data*/



             final int targetWidth = 350;
             final int targetHeight = 350;

            // Load the full-sized image from the URI
            Bitmap fullSizedImageBitmap = null;
            try {
                fullSizedImageBitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), selectedImageUri);
            } catch (IOException e) {
                e.printStackTrace();
            }

            if (fullSizedImageBitmap != null) {
                // Resize the loaded image to a thumbnail size
//                Bitmap thumbnailImageBitmap = ThumbnailUtils.extractThumbnail(fullSizedImageBitmap, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT);

                Bitmap thumbnailImageBitmap = resizeImage(fullSizedImageBitmap, targetWidth, targetHeight);
//                Bitmap thumbnailImageBitmap=fullSizedImageBitmap;
                // Assign the resized thumbnail image to imgdata
                imgdata = new Intent();
                imgdata.putExtra("data", thumbnailImageBitmap);

                // Display the thumbnail image in the ImageView
                imgCamera.setImageBitmap(thumbnailImageBitmap);
            } else {
                Toast.makeText(MainActivity.this, "Error loading image", Toast.LENGTH_SHORT).show();
            }


        }
    }





    // Custom method to resize an image while maintaining aspect ratio
    private Bitmap resizeImage(Bitmap bitmap, int targetWidth, int targetHeight) {
        int originalWidth = bitmap.getWidth();
        int originalHeight = bitmap.getHeight();

        float scaleFactor = Math.min((float) targetWidth / originalWidth, (float) targetHeight / originalHeight);

        Matrix matrix = new Matrix();
        matrix.postScale(scaleFactor, scaleFactor);

        return Bitmap.createBitmap(bitmap, 0, 0, originalWidth, originalHeight, matrix, true);
    }







    // Method to send the captured image to the server
    private void sendImageToServer(Bitmap imageBitmap) {

//        Toast.makeText(MainActivity.this,"called",Toast.LENGTH_SHORT).show();

        // Instantiate the RequestQueue.
        RequestQueue queue = Volley.newRequestQueue(this);

        // Convert the Bitmap image to a Base64-encoded string
        String base64Image = encodeBitmapToBase64(imageBitmap);

        // Create the POST request
        StringRequest stringRequest = new StringRequest(Request.Method.POST, url,
                new Response.Listener<String>() {
                    @Override
                    public void onResponse(String response) {
                        // Handle the server response here
                        output_disp.setText("Prediction: "+response);
                    }
                },
                new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        // Handle errors here
                        Toast.makeText(MainActivity.this,error.getMessage(),Toast.LENGTH_SHORT).show();
                    }
                }) {
            // Set the image data as a POST parameter
            @Override
            protected Map<String, String> getParams() {
                Map<String, String> params = new HashMap<>();
                params.put("image", base64Image);
                return params;
            }
        };

        // Add the request to the RequestQueue.
        queue.add(stringRequest);
    }

    // Method to encode a Bitmap to Base64
    private String encodeBitmapToBase64(Bitmap imageBitmap) {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();
        imageBitmap.compress(Bitmap.CompressFormat.PNG, 100, byteArrayOutputStream);
        byte[] byteArray = byteArrayOutputStream.toByteArray();
        return Base64.encodeToString(byteArray, Base64.DEFAULT);
    }
}