package com.example.diabetic_retinopathy;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;

public class SplashScreen extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_splash_screen);
        Intent iHome = new Intent(SplashScreen.this,MainActivity.class);
        //we run parallel thread (Asynchronous Programming)
        //Handler execute the thread
        new Handler().postDelayed(new Runnable() { //creating object of Runnable class
            @Override
            public void run() {
                startActivity(iHome);
                finish();  //pop the activity from the stack
            }
        },4000);
    }
}